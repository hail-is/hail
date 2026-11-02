package is.hail.collection

// A value paired with an index, e.g. loop state for `tailRecM`.
// Unlike `(Int, A)`, it doesn't box the index.
final class Indexed[A](val i: Int, val value: A)
