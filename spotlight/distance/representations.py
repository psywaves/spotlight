"""
Classes defining user and item latent representations in
factorization models.
"""

import torch.nn as nn

from spotlight.layers import DistanceEmbedding


class CML(nn.Module):
    """
    Collaborative Metric Learning representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the negative of the euclidean
    distance of the item and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 user_embedding_layer=None, item_embedding_layer=None, sparse=False):

        super(CML, self).__init__()

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = DistanceEmbedding(num_users, embedding_dim,
                                                     max_norm=1, sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = DistanceEmbedding(num_items, embedding_dim,
                                                     max_norm=1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        distance = ((user_embedding * item_embedding)**2).sum(1)

        return (-distance)
