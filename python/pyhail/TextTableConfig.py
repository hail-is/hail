
from pyhail.java import scala_object

class TextTableConfig:
    def __init__(self, noheader = False, impute = False,
                 comment = None, delimiter = "\t", missing = "NA", types = None):
        self.noheader = noheader
        self.impute = impute
        self.comment = comment
        self.delimiter = delimiter
        self.missing = missing
        self.types = types

    def asString(self):
        res = ["--comment", self.comment, "--delimiter", self.delimiter,
               "--missing", self.missing]

        if self.noheader:
            res.append("--no-header")

        if self.impute:
            res.append("--impute")

        return " ".join(res)

    def asJavaObject(self, hc):
        return hc.jvm.org.broadinstitute.hail.utils.TextTableConfiguration.apply(self.types, self.comment,
                                                             self.delimiter, self.missing,
                                                             self.noheader, self.impute)

