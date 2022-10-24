# pylint: disable=invalid-name

"""
QualityAssessment data model.
"""


class QualityAssessment:
    """Quality assessment

    :param origin: str, name of the origin function
    :param data: dict, data containing standard fields
    :param context: str, context of QualityAssessment e.g. "Cycle 5"

    """

    # pylint: disable=too-few-public-methods
    def __init__(self, origin=None, data=None, context=None):
        """QualityAssessment data class initialisation"""
        self.origin = origin
        self.data = data
        self.context = context

    def __str__(self):
        """Default printer for QualityAssessment"""
        s = "Quality assessment:\n"
        s += f"\tOrigin: {self.origin}\n"
        s += f"\tContext: {self.context}\n"
        s += "\tData:\n"
        for dataname in self.data.keys():
            s += f"\t\t{dataname}: {str(self.data[dataname])}\n"
        return s
