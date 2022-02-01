from collections import defaultdict
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from aif360.metrics.mdss.ScoringFunctions import Bernoulli, ScoringFunction
from aif360.metrics.mdss.MDSS import MDSS

import pandas as pd

class MDSSClassificationMetric(ClassificationMetric):
    """Bias subset scanning is proposed as a technique to identify bias in
    predictive models using subset scanning [#zhang16]_.

    This class is a wrapper for the bias scan scoring and scanning methods that
    uses the ClassificationMetric abstraction.

    References:
        .. [#zhang16] `Zhang, Z. and Neill, D. B., "Identifying significant
           predictive bias in classifiers," arXiv preprint, 2016.
           <https://arxiv.org/abs/1611.08292>`_
    """
    def __init__(self,
                 dataset: BinaryLabelDataset,
                 classified_dataset: BinaryLabelDataset,
                 scoring_function: ScoringFunction = Bernoulli(direction=None),
                 unprivileged_groups: dict = None,
                 privileged_groups: dict = None):
        """
        Args:
            dataset (BinaryLabelDataset): Dataset containing ground-truth
                labels.
            classified_dataset (BinaryLabelDataset): Dataset containing
                predictions.
            scoring_function (ScoringFunction): Scoring function for MDSS. Note:
                `direction` will be overridden.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.
        """

        super(MDSSClassificationMetric, self).__init__(
            dataset, classified_dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        self.scanner = MDSS(scoring_function)

    def score_groups(self, privileged=True, penalty=1e-17):
        """Compute the bias score for a prespecified group of records.

        Args:
            privileged (bool): Flag for which direction to scan: privileged
                (``True``) implies negative (observed worse than predicted
                outcomes) while unprivileged (``False``) implies positive
                (observed better than predicted outcomes).
            penalty (float): Penalty term. Should be positive. The higher the
                penalty, the less complex (number of features and feature
                values) the highest scoring subset that gets returned is.

        Returns:
            float: Bias score for the given group.
        """
        groups = self.privileged_groups if privileged else self.unprivileged_groups
        subset = dict()

        for g in groups:
            for k, v in g.items():
                if k in subset.keys():
                    subset[k].append(v)
                else:
                    subset[k] = [v]

        coordinates = pd.DataFrame(self.dataset.features, columns=self.dataset.feature_names)
        expected = pd.Series(self.classified_dataset.scores.flatten())
        outcomes = pd.Series(self.dataset.labels.flatten() == self.dataset.favorable_label, dtype=int)

        direction = 'negative' if privileged else 'positive'
        self.scanner.scoring_function.kwargs['direction'] = direction
        return self.scanner.score_current_subset(coordinates, expected, outcomes, dict(subset), penalty)

    def bias_scan(self, privileged=True, num_iters=10, penalty=1e-17):
        """Scan to find the highest scoring subset of records.

        Args:
            privileged (bool): Flag for which direction to scan: privileged
                (``True``) implies negative (observed worse than predicted
                outcomes) while unprivileged (``False``) implies positive
                (observed better than predicted outcomes).
            num_iters (int): Number of iterations (random restarts).
            penalty (float): Penalty coefficient. Should be positive. The higher
                the penalty, the less complex (number of features and feature
                values) the highest scoring subset that gets returned is.

        Returns:
            tuple:
                Highest scoring subset and its bias score

                * **subset** (dict) -- Mapping of feature to value defining the
                highest scoring subset.
                * **score** (float) -- Bias score for that group.
        """
        coordinates = pd.DataFrame(self.classified_dataset.features, columns=self.classified_dataset.feature_names)
        expected = pd.Series(self.classified_dataset.scores.flatten())
        outcomes = pd.Series(self.dataset.labels.flatten() == self.dataset.favorable_label, dtype=int)

        direction = 'negative' if privileged else 'positive'
        self.scanner.scoring_function.kwargs['direction'] = direction
        return self.scanner.scan(coordinates, expected, outcomes, penalty, num_iters)