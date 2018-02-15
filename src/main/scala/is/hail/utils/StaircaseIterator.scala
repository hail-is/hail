package is.hail.utils

class StaircaseIterator[A](it: EphemeralIterator[A], equiv: EquivalenceClassView[A])
  extends StagingIterator[EphemeralIterator[A]] {

  equiv.setEmpty()
  var isValid = true

  object stepIterator extends EphemeralIterator[A] {
    def head = it.value
    def isValid = it.isValid && equiv.inEquivClass(head)
    def advanceHead() { it.advance() }
  }
  advanceHead()

  def head = stepIterator
  def advanceHead() {
    exhaustStep()
    if (it.isValid)
      equiv.setEquivClass(it.value)
    else {
      equiv.setEmpty()
      isValid = false
    }
  }

  private def exhaustStep() {
    while (head.isValid) head.advance()
  }
}
