package is.hail.cxx

object Code {
  def apply(args: Code*): Code = sequence(args)

  def sequence(codes: Seq[Code]): Code = codes.mkString("\n")

  def defineVars(variables: Seq[Variable]): Code = sequence(variables.map(_.define))
}
