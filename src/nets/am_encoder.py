"""AM graph encoder."""
from torch import nn
from nets.graph_encoder import MultiHeadAttentionLayer
class GraphAttentionEncoder(nn.Module):

    def __init__(self, n_heads, embed_dim, n_layers, node_dim=None, normalization='batch', feed_forward_hidden=512, prompt_dim=0):
        super(GraphAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.layers = nn.Sequential(*(MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization) for _ in range(n_layers)))

    def forward(self, x, prompt, mask=None):
        assert mask is None, 'TODO mask not yet supported!'
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h = self.layers(h)
        return (h[:, :-1, :], h.mean(dim=1), h[:, -2:-1, :])
