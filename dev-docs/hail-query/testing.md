# Writing Tests in Hail

Tests live in `hail/test/src/is/hail/` and run on JUnit 5. A global extension (`HailExtension`) manages all Hail-specific state — Spark, the filesystem, region pools, execution contexts — so test classes themselves need no base class or boilerplate.

## State Management

`HailExtension` is registered globally via `META-INF/services` and manages four tiers of state:

| Scope | Lifetime | Contents |
|-------|----------|----------|
| Test run | Entire test run | `SparkBackend`, `HailClassLoader`, `HailFeatureFlags`, `FS`, `RegionPool`, reference genomes (`SharedResources`) |
| Test method | One `@Test` method | `ExecuteContext` — with its own `Region`, `ExecutionTimer`, `OwningTempFileManager`, `IrMetadata`. Shared across all parameterized invocations of that method, and between a `@BeforeEach` and its following `@Test` |
| Fallback | Class scope | An `ExecuteContext` created when no test method is in scope (e.g. from `@BeforeAll`) |

`SharedResources` is **lazy**: it is created on the first parameter injection that needs it. A test class that never declares any Hail-state parameters never triggers Spark startup.

When an `ExecuteContext` is created, it references the long-lived `SharedResources` for backend/fs/pool/classloader/flags/references, points at the class-scoped IR caches, and gets a fresh Region/timer/tempFileManager/irMetadata. JUnit automatically closes it (releasing the Region, finishing the timer, cleaning temp files) when its owning scope ends.

The `ExecuteContext` is given no-op maps for all caches such as the compile cache, to ensure test isolation.

## Parameter Injection

Declare parameters on test methods and JUnit will resolve them through `HailExtension`. Supported types:

- `ExecuteContext` — scoped as described above
- `FS`
- `Backend`
- `RegionPool`

`FS`, `Backend`, and `RegionPool` resolve directly to the shared singleton objects.

Declare `ExecuteContext` as `implicit` so that `JUnitTestUtils` helpers (which take `implicit ctx: ExecuteContext`) pick it up automatically:

```scala
import is.hail.JUnitTestUtils._

class MySuite {
  @Test def testSomething(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(I32(5), 5)
}
```

This works because the JVM sees Scala's implicit parameter as a plain parameter, so JUnit's resolver handles it normally.

`@BeforeEach` methods can also declare injected parameters. The `ExecuteContext` created for `@BeforeEach` is reused by the following `@Test` method (they share a single Region):

```scala
class MySuite {
  @BeforeEach def setUp(implicit ctx: ExecuteContext): Unit = { /* ... */ }
  @Test def testFoo(implicit ctx: ExecuteContext): Unit = { /* same ctx as setUp */ }
}
```

## Parameterized Tests

Use `@is.hail.ParameterizedTest` (not JUnit's `@org.junit.jupiter.params.ParameterizedTest`) to run a test method once per element from a factory method.

**Same-name factory (default):** bare `@ParameterizedTest`, factory shares the test's name:

```scala
import is.hail.ParameterizedTest

class MySuite {
  def testAddition = Seq((1, 2, 3), (0, 0, 0), (-1, 1, 0))

  @ParameterizedTest def testAddition(a: Int, b: Int, expected: Int): Unit =
    assertEq(a + b, expected)
}
```

**Named factory:** `@ParameterizedTest("factoryName")`:

```scala
class MySuite {
  def sharedData = Seq(1, 2, 3)

  @ParameterizedTest("sharedData") def testPositive(n: Int): Unit = assert(n > 0)
  @ParameterizedTest("sharedData") def testNonZero(n: Int): Unit = assert(n != 0)
}
```

How it works (`HailMethodArgumentsProvider`):

- Looks up the factory method by name on the test class.
- Invokes it through JUnit's `ExecutableInvoker`, so factory parameters go through the same resolver chain — factories can declare `ctx`, `fs`, etc.
- Scala tuples are splatted into positional test-method parameters. Everything else is passed as a single argument.
- The factory can return any Scala or Java collection type (`Seq`, `IndexedSeq`, `Iterator`, `Set`, `Array`, `Stream`, `Iterable`, ...).
- Extension-resolved parameters (like `ctx`) come **after** the factory-provided parameters. Put `ctx` in its own implicit parameter list:

```scala
def testFoo = Seq(("a", 1), ("b", 2))

@ParameterizedTest def testFoo(s: String, n: Int)(implicit ctx: ExecuteContext): Unit = ...
```

## Cheat Sheet

**Simple test:**
```scala
import org.junit.jupiter.api.Test

class FooSuite {
  @Test def testBar(): Unit = assert(1 + 1 == 2)
}
```

**Test with ExecuteContext:**
```scala
import is.hail.JUnitTestUtils._
import org.junit.jupiter.api.Test

class FooSuite {
  @Test def testBar(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(I32(42), 42)
}
```

**IR tests with execution strategies:**
```scala
import is.hail.ExecStrategy
import is.hail.ExecStrategy.ExecStrategy
import is.hail.JUnitTestUtils._
import org.junit.jupiter.api.Test

class FooSuite {
  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.nonLowering

  @Test def testBar(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(I32(42), 42) // runs under each strategy in execStrats

  @Test def testBaz(implicit ctx: ExecuteContext): Unit = {
    implicit val execStrats = ExecStrategy.lowering // override for this test
    assertEvalsTo(I32(42), 42)
  }
}
```

**Parameterized test:**
```scala
import is.hail.ParameterizedTest

class FooSuite {
  def testBar = Seq((1, 1), (2, 4), (3, 9))

  @ParameterizedTest def testBar(n: Int, expected: Int): Unit =
    assertEq(n * n, expected)
}
```

**Setup/teardown:**
```scala
import org.junit.jupiter.api.{BeforeEach, AfterEach, BeforeAll, AfterAll, Test}

class FooSuite {
  @BeforeAll  def classSetUp(): Unit = { /* once before all tests */ }
  @AfterAll   def classTearDown(): Unit = { /* once after all tests */ }
  @BeforeEach def setUp(): Unit = { /* before each test */ }
  @AfterEach  def tearDown(): Unit = { /* after each test */ }
}
```

**Key helpers in `JUnitTestUtils`:**

| Helper | Purpose |
|--------|---------|
| `assertEq(actual, expected)` | Equality check with Scala-conventional argument order |
| `assertEvalsTo(ir, expected)` | Evaluate IR under each `ExecStrategy` and check result |
| `assertAllEvalTo((ir1, v1), (ir2, v2), ...)` | Batch IR evaluation |
| `assertFatal(ir, regex)` | Assert IR evaluation throws `HailException` matching regex |
| `intercept[E] { ... }` | Assert block throws `E`, return the exception |
| `interceptFatal(regex) { ... }` | Assert block throws `HailException` with message matching regex |
| `check(prop)` | Run a ScalaCheck property with deterministic seed |
| `importVCF(path, ...)` | Build a `MatrixIR` from a VCF file |
| `getTestResource(path)` | Resolve a path relative to the test resources directory |

**Things to avoid:**
- Don't call JUnit's `assertEquals(expected, actual)` directly — use `assertEq(actual, expected)` to avoid argument-order confusion.
- Don't write `@ExtendWith(classOf[HailExtension])` — the extension is globally registered.
- Don't add `@TestInstance(Lifecycle.PER_CLASS)` — it's forced globally via `junit-platform.properties`.
