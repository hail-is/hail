package is.hail.utils

case class SemanticVersion(major: Int, minor: Int, patch: Int) extends Ordered[SemanticVersion] {
  assert((major & 0xff) == major)
  assert((minor & 0xff) == minor)
  assert((patch & 0xff) == patch)

  def supports(that: SemanticVersion): Boolean =
    major == that.major &&
      that.minor <= minor

  def rep: Int = (major << 16) | (minor << 8) | patch

  override def toString: String = s"$major.$minor.$patch"

  override def compare(that: SemanticVersion): Int = {
    if (major != that.major)
      Integer.compare(major, that.major)
    else if (minor != that.minor)
      Integer.compare(minor, that.minor)
    else
      Integer.compare(patch, that.patch)
  }
}

object SemanticVersion {
  def apply(rep: Int): SemanticVersion =
    SemanticVersion((rep >> 16) & 0xff, (rep >> 8) & 0xff, rep & 0xff)
}
