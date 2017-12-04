package is.hail.annotations.ordering;

import is.hail.expr.*;
import is.hail.annotations.*;
import is.hail.variant.Contig;

public class StaticContigUnsafeOrdering {
  public static int contigCompare(String l, String r) {
    return Contig.compare(l, r);
  }
}
