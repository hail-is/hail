package is.hail.cxx

object Code {
  def apply(args: Code*): Code = sequence(args)

  def sequence(codes: Seq[Code]): Code = codes.mkString("\n")
}
