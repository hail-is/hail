package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.expr.MatrixLocalValue
import is.hail.expr.types._

object MatrixFileMetadata {
  def apply(matrixType: MatrixType, localValue: MatrixLocalValue): MatrixFileMetadata = MatrixFileMetadata(matrixType, localValue, None)

  def apply(sampleAnnotations: IndexedSeq[Annotation],
    globalAnnotation: Annotation = Annotation.empty,
    saSignature: TStruct = TStruct("s" -> TString()),
    colKey: IndexedSeq[String] = Array("s"),
    vSignature: Type = TVariant(GenomeReference.defaultReference),
    vaSignature: TStruct = TStruct.empty(),
    globalSignature: TStruct = TStruct.empty(),
    // FIXME require
    genotypeSignature: TStruct = Genotype.htsGenotypeType): MatrixFileMetadata = {
    MatrixFileMetadata(
      MatrixType(globalSignature, saSignature, colKey, vSignature, vaSignature, genotypeSignature),
      MatrixLocalValue(globalAnnotation, sampleAnnotations))
  }
}

case class MatrixFileMetadata(
  matrixType: MatrixType,
  localValue: MatrixLocalValue,
  partitionCounts: Option[Array[Long]]) {
}
