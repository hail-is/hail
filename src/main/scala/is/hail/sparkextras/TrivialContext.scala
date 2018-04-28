package is.hail.sparkextras

object TrivialContextInstance extends AutoCloseable {
  def close(): Unit = ()
}
