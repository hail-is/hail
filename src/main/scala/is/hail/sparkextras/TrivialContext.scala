package is.hail.sparkextras

object TrivialContextInstance extends ResettableContext {
  def close(): Unit = ()

  def reset(): Unit = ()
}
