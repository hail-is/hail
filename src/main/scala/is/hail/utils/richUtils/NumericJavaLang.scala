package is.hail.utils.richUtils

object NumericJavaLang {

  object NumericJavaLangInt extends Numeric[java.lang.Integer] {

    def plus(x: java.lang.Integer, y: java.lang.Integer): java.lang.Integer = new java.lang.Integer(x.intValue() + y.intValue())

    def minus(x: java.lang.Integer, y: java.lang.Integer): java.lang.Integer = new java.lang.Integer(x.intValue() - y.intValue())

    def times(x: java.lang.Integer, y: java.lang.Integer): java.lang.Integer = new java.lang.Integer(x.intValue() * y.intValue())

    def negate(x: java.lang.Integer): java.lang.Integer = new java.lang.Integer(-x.intValue())

    def fromInt(x: Int): java.lang.Integer = new java.lang.Integer(x)

    def toInt(x: java.lang.Integer): Int = x.intValue()

    def toLong(x: java.lang.Integer): Long = x.longValue()

    def toFloat(x: java.lang.Integer): Float = x.floatValue()

    def toDouble(x: java.lang.Integer): Double = x.doubleValue()

    def compare(x: java.lang.Integer, y: java.lang.Integer): Int = x.intValue() - y.intValue()
  }

  object NumericJavaLangLong extends Numeric[java.lang.Long] {

    def plus(x: java.lang.Long, y: java.lang.Long): java.lang.Long = new java.lang.Long(x.longValue() + y.longValue())

    def minus(x: java.lang.Long, y: java.lang.Long): java.lang.Long = new java.lang.Long(x.longValue() - y.longValue())

    def times(x: java.lang.Long, y: java.lang.Long): java.lang.Long = new java.lang.Long(x.longValue() * y.longValue())

    def negate(x: java.lang.Long): java.lang.Long = new java.lang.Long(-x.longValue())

    def fromInt(x: Int): java.lang.Long = new java.lang.Long(x)

    def toInt(x: java.lang.Long): Int = x.intValue()

    def toLong(x: java.lang.Long): Long = x.longValue()

    def toFloat(x: java.lang.Long): Float = x.floatValue()

    def toDouble(x: java.lang.Long): Double = x.doubleValue()

    def compare(x: java.lang.Long, y: java.lang.Long): Int = {
      val diff = x.longValue() - y.longValue()
      if (diff > 0) 1 else if (diff < 0) -1 else 0
    }
  }

  object NumericJavaLangFloat extends Numeric[java.lang.Float] {

    def plus(x: java.lang.Float, y: java.lang.Float): java.lang.Float = new java.lang.Float(x.floatValue() + y.floatValue())

    def minus(x: java.lang.Float, y: java.lang.Float): java.lang.Float = new java.lang.Float(x.floatValue() - y.floatValue())

    def times(x: java.lang.Float, y: java.lang.Float): java.lang.Float = new java.lang.Float(x.floatValue() * y.floatValue())

    def negate(x: java.lang.Float): java.lang.Float = new java.lang.Float(-x.floatValue())

    def fromInt(x: Int): java.lang.Float = new java.lang.Float(x)

    def toInt(x: java.lang.Float): Int = x.intValue()

    def toLong(x: java.lang.Float): Long = x.longValue()

    def toFloat(x: java.lang.Float): Float = x.floatValue()

    def toDouble(x: java.lang.Float): Double = x.doubleValue()

    def compare(x: java.lang.Float, y: java.lang.Float): Int = {
      val diff = x.floatValue() - y.floatValue()
      if (diff > 0) 1 else if (diff < 0) -1 else 0
    }
  }

  object NumericJavaLangDouble extends Numeric[java.lang.Double] {

    def plus(x: java.lang.Double, y: java.lang.Double): java.lang.Double = new java.lang.Double(x.doubleValue() + y.doubleValue())

    def minus(x: java.lang.Double, y: java.lang.Double): java.lang.Double = new java.lang.Double(x.doubleValue() - y.doubleValue())

    def times(x: java.lang.Double, y: java.lang.Double): java.lang.Double = new java.lang.Double(x.doubleValue() * y.doubleValue())

    def negate(x: java.lang.Double): java.lang.Double = new java.lang.Double(-x.doubleValue())

    def fromInt(x: Int): java.lang.Double = new java.lang.Double(x)

    def toInt(x: java.lang.Double): Int = x.intValue()

    def toLong(x: java.lang.Double): Long = x.longValue()

    def toFloat(x: java.lang.Double): Float = x.floatValue()

    def toDouble(x: java.lang.Double): Double = x.doubleValue()

    def compare(x: java.lang.Double, y: java.lang.Double): Int = {
      val diff = x.doubleValue() - y.doubleValue()
      if (diff > 0) 1 else if (diff < 0) -1 else 0
    }
  }
}