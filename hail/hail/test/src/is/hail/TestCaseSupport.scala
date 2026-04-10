package is.hail

trait TestCaseSupport { self: munit.FunSuite =>
  trait TestCases {
    var i: Int = 0

    def test(name: String)(body: => Any)(implicit loc: munit.Location): Unit = {
      i += 1
      self.test(s"$name case $i")(body)
    }
  }
}
