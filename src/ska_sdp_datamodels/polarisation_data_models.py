# pylint: disable=invalid-name

"""
Data models specifically used for Polarisation
"""

__all__ = ["ReceptorFrame", "PolarisationFrame"]


class ReceptorFrame:
    """Polarisation frames for receptors

    This includes circular, linear, and stokesI. The latter is
    non-physical but useful for some types of testing.

    Attributes:
        rec_frames: Dictionary containing all the supported polarisations

    """

    rec_frames = {
        "circular": {"R": 0, "L": 1},
        "circularnp": {"R": 0, "L": 1},
        "linear": {"X": 0, "Y": 1},
        "linearnp": {"X": 0, "Y": 1},
        "stokesI": {"I": 0},
    }

    def __init__(self, name):
        """create ReceptorFrame

        :param name: Name of Polarisation
        """

        if name in self.rec_frames:
            self.type = name
            self.translations = self.rec_frames[name]
        else:
            raise ValueError(f"Unknown receptor frame {str(name)}")

    def __eq__(self, a):
        return self.type == a.type

    def valid(self, name):
        """Is name a valid rec_frame key?"""
        return name in self.rec_frames

    @property
    def nrec(self):
        """Number of receptors (should be 2)"""
        return len(list(self.translations.keys()))

    @property
    def names(self):
        """Names"""
        return list(self.translations.keys())


class PolarisationFrame:
    """
    Polarisation Frame data class

    Defines polarisation frames post correlation: include
    stokesI, stokesIQUV, linear, circular.

    TODO: Check if this is correct
    Attributes:
        fits_codes: PolarisationFrames that fit
                    the data dimensions in actual coordinates
        polarisation_frames: Dictionary containing
                    the source reference frames information
        fits_to_rascil: PolarisationFrames that fit
                    the data dimensions in the code

    """

    fits_codes = {
        "circular": [-1, -2, -3, -4],  # RR, LL, RL, LR
        "circularnp": [-1, -2],  # RR, LL,
        "linear": [-5, -6, -7, -8],  # XX, YY, XY, YX
        "linearnp": [-5, -6],  # XX, YY
        "stokesIQUV": [1, 2, 3, 4],  # I, Q, U, V
        "stokesIV": [1, 4],  # IV
        "stokesIQ": [1, 2],  # IQ
        "stokesI": [1],  # I
    }
    polarisation_frames = {
        "circular": {"RR": 0, "RL": 1, "LR": 2, "LL": 3},
        "circularnp": {"RR": 0, "LL": 1},
        "linear": {"XX": 0, "XY": 1, "YX": 2, "YY": 3},
        "linearnp": {"XX": 0, "YY": 1},
        "stokesIQUV": {"I": 0, "Q": 1, "U": 2, "V": 3},
        "stokesIV": {"I": 0, "V": 1},
        "stokesIQ": {"I": 0, "Q": 1},
        "stokesI": {"I": 0},
    }
    fits_to_rascil = {
        "circular": [0, 3, 1, 2],  # RR, LL, RL, LR
        "circularnp": [0, 1],  # RR, LL,
        "linear": [0, 3, 1, 2],  # XX, YY, XY, YX
        "linearnp": [0, 1],  # XX, YY
        "stokesIQUV": [0, 1, 2, 3],  # I, Q, U, V
        "stokesIV": [0, 1],  # IV
        "stokesIQ": [0, 1],  # IQ
        "stokesI": [0],  # I
    }

    def __init__(self, name):
        """create PolarisationFrame

        :param name: Name of Polarisation
        """

        if name in self.polarisation_frames:
            self.type = name
            self.translations = self.polarisation_frames[name]
        else:
            raise ValueError(f"Unknown polarisation frame {name}")

    def __eq__(self, a):
        if a is None:
            return False
        return self.type == a.type

    def __str__(self):
        """Default printer for Polarisation"""
        return self.type

    @property
    def npol(self):
        """Number of correlated polarisations"""
        return len(list(self.translations.keys()))

    @property
    def names(self):
        """Names"""
        return list(self.translations.keys())
