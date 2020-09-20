# Some of the strategies are adopted directly from Github
from .wasserstein_adversarial import WAAL
from .entropy import Entropy
from .random import Random
from .waal_no_semi import SWAAL
from .entropy_sampling import EntropySampling
from .entropy_sampling import Semi_EntropySampling
from .waal_with_fixmatch import WAALFixMatch
from .waaluncertainty import WAALUncertainty
from .farthest_first import FarthestFirst
from .entropy_with_fixmatch import FixMatchEntropy
from .random_with_fixmatch import FixMatchRandom
from .self_training_entropy import EntropySelfTraining
from .farthest_first_entropy import FarthestFirstEntropy
from .discriminative_representation_sampling import DiscriminativeRepresentationSampling
