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