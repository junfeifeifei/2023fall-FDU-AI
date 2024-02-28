import torch
import torch.nn as nn


def log_sum_exp(vec):
    # 计算 log-sum-exp 操作，用于数值稳定性
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))


class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab, label_map, device='cpu'):
        """
       双向长短时记忆网络 (BiLSTM) 和条件随机场 (CRF) 结合的模型

       Args:
       - embedding_dim: 词嵌入维度
       - hidden_dim: 隐藏层维度
       - vocab: 词汇表
       - label_map: 标签映射
       - device: 设备 ('cpu' 或 'cuda')
       """
        super(BiLSTM_CRF, self).__init__()
        # 嵌入维度
        self.embedding_dim = embedding_dim
        # 隐藏层维度
        self.hidden_dim = hidden_dim
        self.vocab_size = len(vocab)
        self.tagset_size = len(label_map)
        self.device = device
        self.state = 'train'
        self.crf = CRF(label_map, device)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=True)

    def extract_lstm_features(self, sentence, seq_len):
        # 获取 BiLSTM-CRF 模型的特征
        """
       提取BiLSTM的特征

       Args:
       - sentence: 输入句子
       - seq_len: 句子的实际长度

       Returns:
       - torch.Tensor: BiLSTM的特征
        """
        embeds = self.word_embeds(sentence)
        self.dropout(embeds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        seqence_output = self.layer_norm(seq_unpacked)
        lstm_feats = self.hidden2tag(seqence_output)
        return lstm_feats

    def forward(self, sentence, seq_len, tags=''):
        """
        模型前向传播

        Args:
        - sentence: 输入句子
        - seq_len: 句子的实际长度
        - tags: 真实标签（用于训练）

        Returns:
        - torch.Tensor or List[List[int]]: 训练时返回损失，评估时返回预测标签
        """
        feats = self.extract_lstm_features(sentence, seq_len)
        if self.state == 'train':
            # 计算损失（用于训练）
            loss = self.crf.calculate_negative_log_likelihood(feats, tags, seq_len)
            return loss
        elif self.state == 'eval':
            # 在评估模式下，进行 Viterbi 解码获取预测标签序列
            all_tag = []
            for i, feat in enumerate(feats):
                all_tag.append(self.crf.decode_viterbi(feat[:seq_len[i]])[1])
            return all_tag
        else:
            # 在其他状态下，进行 Viterbi 解码获取预测标签序列
            return self.crf.decode_viterbi(feats[0])[1]


class CRF(nn.Module):
    def __init__(self, label_map, device='cpu'):
        """
        条件随机场 (CRF) 模型

        Args:
        - label_map: 标签映射
        - device: 设备 ('cpu' 或 'cuda')
        """
        super(CRF, self).__init__()
        self.label_map = label_map
        # 标签的索引映射回标签
        self.label_map_inv = {v: k for k, v in label_map.items()}
        self.tagset_size = len(self.label_map)
        self.device = device
        # 创建一个可学习的参数矩阵，表示从一个标签转移到另一个标签的转移概率
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        # 约束起始标签只能出现在序列的开始位置。
        self.transitions.data[self.label_map[self.START_TAG], :] = -10000
        # 约束停止标签只能出现在序列的结束位置。
        self.transitions.data[:, self.label_map[self.STOP_TAG]] = -10000

    def calculate_forward_algorithm(self, feats, seq_len):
        """
        计算前向算法的结果

        Args:
        - feats: BiLSTM的特征
        - seq_len: 句子的实际长度

        Returns:
        - torch.Tensor: 前向算法的结果
        """
        # 前向算法计算分数
        init_alphas = torch.full((self.tagset_size,), -10000.)
        init_alphas[self.label_map[self.START_TAG]] = 0.
        forward_var = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32,
                                  device=self.device)
        forward_var[:, 0, :] = init_alphas
        transitions = self.transitions.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        for seq_i in range(feats.shape[1]):
            emit_score = feats[:, seq_i, :]
            tag_var = (
                    forward_var[:, seq_i, :].unsqueeze(1).repeat(1, feats.shape[2], 1)
                    + transitions
                    + emit_score.unsqueeze(2).repeat(1, 1, feats.shape[2])
            )
            cloned = forward_var.clone()
            cloned[:, seq_i + 1, :] = log_sum_exp(tag_var)
            forward_var = cloned
        forward_var = forward_var[range(feats.shape[0]), seq_len, :]
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]].unsqueeze(0).repeat(feats.shape[0],
                                                                                                         1)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def get_sentence_score(self, feats, tags, seq_len):
        """
        计算给定标签序列的分数

        Args:
        - feats: BiLSTM的特征
        - tags: 真实标签
        - seq_len: 句子的实际长度

        Returns:
        - torch.Tensor: 标签序列的分数
        """
        # 计算给定标签序列的分数
        score = torch.zeros(feats.shape[0], device=self.device)
        start = torch.tensor([self.label_map[self.START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0],
                                                                                                       1)
        tags = torch.cat([start, tags], dim=1)
        for batch_i in range(feats.shape[0]):
            # tags 前一个标签，后一个标签
            # torch.sum计算发射概率的得分
            score[batch_i] = torch.sum(
                self.transitions[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]]) \
                             + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]])
            score[batch_i] += self.transitions[self.label_map[self.STOP_TAG], tags[batch_i][seq_len[batch_i]]]
        return score

    def decode_viterbi(self, feats):
        """
        使用 Viterbi 算法解码得到最优标签序列

        Args:
        - feats: BiLSTM的特征

        Returns:
        - Tuple[torch.Tensor, List[int]]: 最优路径的分数和标签序列
        """
        # 使用 Viterbi 算法解码得到最优标签序列
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.label_map[self.START_TAG]] = 0
        forward_var = init_vvars
        for feat in feats:
            forward_var = forward_var.repeat(feat.shape[0], 1)
            next_tag_var = forward_var + self.transitions
            bptrs_t = torch.max(next_tag_var, 1)[1].tolist()
            viterbivars_t = next_tag_var[range(forward_var.shape[0]), bptrs_t]
            forward_var = (viterbivars_t + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]]
        best_tag_id = torch.max(terminal_var, 1)[1].item()
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.label_map[self.START_TAG]
        best_path.reverse()
        return path_score, best_path

    def calculate_negative_log_likelihood(self, feats, tags, seq_len):
        # 计算负对数似然损失
        """
       计算负对数似然损失

       Args:
       - feats: BiLSTM的特征
       - tags: 真实标签
       - seq_len: 句子的实际长度

       Returns:
       - torch.Tensor: 负对数似然损失
       """
        forward_score = self.calculate_forward_algorithm(feats, seq_len)
        gold_score = self.get_sentence_score(feats, tags, seq_len)
        return torch.mean(forward_score - gold_score)
