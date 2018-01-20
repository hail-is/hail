package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.expr.MatrixLocalValue
import is.hail.expr.types._

object MatrixFileMetadata {
  def apply(matrixType: MatrixType, localValue: MatrixLocalValue): MatrixFileMetadata = MatrixFileMetadata(matrixType, localValue, None)

  def apply(sampleIds: IndexedSeq[String],
    sampleAnnotations: IndexedSeq[Annotation] = null,
    globalAnnotation: Annotation = Annotation.empty,
    sSignature: Type = TString(),
    saSignature: TStruct = TStruct.empty(),
    vSignature: Type = TVariant(GenomeReference.defaultReference),
    vaSignature: TStruct = TStruct.empty(),
    globalSignature: TStruct = TStruct.empty(),
    // FIXME require
    genotypeSignature: TStruct = Genotype.htsGenotypeType): MatrixFileMetadata = {
    MatrixFileMetadata(
      MatrixType(globalSignature, sSignature, saSignature, vSignature, vaSignature, genotypeSignature),
      MatrixLocalValue(globalAnnotation, sampleIds,
        if (sampleAnnotations == null)
          Annotation.emptyIndexedSeq(sampleIds.length)
        else
          sampleAnnotations))
  }
}

case class MatrixFileMetadata(
  matrixType: MatrixType,
  localValue: MatrixLocalValue,
  partitionCounts: Option[Array[Long]]) {
}
