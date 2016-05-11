package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{TInt, TDouble, TStruct}
import org.broadinstitute.hail.methods.SexCheckPlink
import org.kohsuke.args4j.{Option => Args4jOption}


object SexCheck extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(required = false, name = "-s", aliases = Array("--store"), usage = "Output file")
    var output2: String = _
  }

  //val header = "ID\tOrigSex\tImputedSex\tFlag\n"

  def newOptions = new Options

  def name = "sexcheck"

  def description = "Check gender of samples"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val result = SexCheckPlink.imputeSex(state.vds).result.collect().toMap

    val signature = TStruct("F" -> TDouble, "E" -> TDouble, "O" -> TDouble, "N" -> TInt, "T" -> TInt, "imputedSex" -> TInt)

    val (newSAS, insertSexCheck) = state.vds.saSignature.insert(signature, "sexcheck")
    val newSampleAnnotations = state.vds.sampleIdsAndAnnotations
      .map { case (s, sa) =>
        insertSexCheck(sa, result.get(s))
      }

    val output = options.output

/*    if (output != null) {
      hadoopDelete(output, state.hadoopConf, recursive = true)
      scheck.inbreedingCoefficients.map { case (s, ibc) =>
        val sb = new StringBuilder()
        sb.append(s)
        sb += '\t'
        sb.append(ibc.O)
        sb += '\t'
        sb.append(ibc.E)
        sb += '\t'
        sb.append(ibc.N)
        sb += '\t'
        sb.append(ibc.F)
        sb += '\t'
        sb.append(SexCheckPlink.determineSex(ibc))
        sb += '\t'
        sb.append(ibc.T)
        sb.result()
      }.writeTable(output, Some("SampleID\tO(HOM)\tE(HOM)\tN(NM)\tF\tImputedSex\tT"))*/
 //   }

    //println(imputedSex)
/*    val stats = scheck.inbreedingCoefficients.map
    imputedSex.map{case (s, newgender) =>
      val sb = new StringBuilder()
      sb.append(state.vds.sampleIds(s))
      sb.append("\t")
      sb.append(".") // sample annotation for reported gender
      sb.append("\t")
      sb.append(newgender)
      sb.append("\t")
      sb.append(flag)
      sb.append("\n")
      sb.result()
    }*/
    state
  }
}
