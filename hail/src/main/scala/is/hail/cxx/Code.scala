package is.hail.cxx

object Code {
  def apply(args: Code*): Code =
    args.mkString
}
