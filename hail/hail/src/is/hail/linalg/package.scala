package is.hail

// re-exported so that linear-algebra clients need not depend on breeze
package object linalg {
  type MatrixSingularException = breeze.linalg.MatrixSingularException

  type NotConvergedException = breeze.linalg.NotConvergedException

  val NotConvergedException: breeze.linalg.NotConvergedException.type =
    breeze.linalg.NotConvergedException
}
