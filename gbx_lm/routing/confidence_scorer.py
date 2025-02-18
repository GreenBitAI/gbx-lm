from typing import Union, Tuple
from .libra_router.ue_router import MahalanobisDistanceSeq


class ConfidenceScorer:
    """
    A class to compute confidence scores based on Mahalanobis distance.

    Attributes:
        parameters_path (str): Path to the model parameters
        json_file_path (str): Path to the uncertainty bounds JSON file
        device (str): Device to run computations on ('cpu', 'cuda', 'mps')

    Example:
        confidence_scorer = ConfidenceScore(
            parameters_path="path/to/params",
            json_file_path="path/to/bounds.json",
            device="cuda"
        )

        confidence = confidence_scorer.calculate_confidence(hidden_states)
    """

    def __init__(
            self,
            parameters_path: str,
            model_id: str,
            device: str = "mps"
    ):
        """
        Initialize the ConfidenceScore calculator.

        Args:
            parameters_path: Path to model parameters
            json_file_path: Path to uncertainty bounds JSON
            device: Computation device
            threshold: Confidence threshold for routing
        """
        self.parameters_path = parameters_path
        self.device = device

        # Initialize Mahalanobis distance calculator
        try:
            self.mahalanobis = MahalanobisDistanceSeq(
                parameters_path=parameters_path,
                normalize=False,
                model_id=model_id,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mahalanobis distance calculator: {str(e)}")

    def calculate_confidence(
            self,
            hidden_states: list,
            return_uncertainty: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Calculate confidence score from hidden states.

        Args:
            hidden_states: Model hidden states tensor
            return_uncertainty: Whether to return the raw uncertainty score

        Returns:
            float or tuple: Confidence score (and uncertainty if return_uncertainty=True)

        Raises:
            ValueError: If hidden states have invalid shape
            RuntimeError: If confidence calculation fails
        """

        try:
            # Calculate uncertainty using Mahalanobis distance
            uncertainty = self.mahalanobis(hidden_states)
            if uncertainty is None:
                raise RuntimeError("Failed to calculate uncertainty")

            # Normalize uncertainty if bounds are available
            if self.mahalanobis.ue_bounds_tensor is not None:
                uncertainty = self.mahalanobis.normalize_ue(
                    uncertainty[0],
                    self.device
                )
            else:
                uncertainty = uncertainty[0]

            # Convert uncertainty to confidence score
            confidence_score = 1.0 - float(uncertainty)

            if return_uncertainty:
                return confidence_score, float(uncertainty)
            return confidence_score

        except Exception as e:
            raise RuntimeError(f"Failed to calculate confidence score: {str(e)}")