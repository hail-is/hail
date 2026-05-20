package is.hail

import is.hail.annotations.{Region, RegionPool}
import is.hail.backend.{Backend, ExecuteContext}
import is.hail.io.fs.FS

import scala.collection.mutable

import org.junit.jupiter.api.{BeforeEach, Test, TestInfo}
import org.junit.jupiter.api.Assertions.{assertSame, assertTrue}

/** Verifies that the resource lifetimes `HailExtension` hands to tests match the design in
  * `dev-docs/hail-query/testing.md`.
  *
  * Properties tested here:
  *   - `ctx.fs` / `ctx.backend` / `ctx.r.pool` are the same shared instances that the resolver
  *     hands to `FS` / `Backend` / `RegionPool` parameters (no wrapping) and are the same across
  *     every test method.
  *   - `@BeforeEach` and the following `@Test` receive the same `ExecuteContext`.
  *   - Two `ExecuteContext` parameters on one method resolve to the same instance.
  *   - Each `@ParameterizedTest` invocation shares one `ExecuteContext` (and `Region`) with the
  *     factory that produced its arguments and with every other invocation.
  *   - Different test methods receive distinct per-method state: `ExecuteContext`, `Region`,
  *     `ExecutionTimer`, `tempFileManager`, `irMetadata`.
  *   - The class-level IR caches (`BlockMatrixCache`, `CompileCache`, `PersistedIrCache`,
  *     `PersistedCoercerCache`) are the same mutable-map instances across every test method.
  *
  * Properties the design requires that this suite does NOT test, and why:
  *   - `SharedResources` lazy init. Any suite that resolves a Hail parameter has already triggered
  *     creation, so a test can't observe "not yet initialized" from within itself. A cold-start
  *     observation would need external inspection of the process before the first injected test
  *     runs.
  *   - Teardown ordering (Spark stopped before the `RegionPool`) in `SharedResources.close`: that
  *     runs after the last test; no test method is alive to observe it.
  *   - `@BeforeAll`-scoped ctx living at the class scope (design decision 9). Testing it would
  *     require `@BeforeAll def setup(ctx: ExecuteContext)`, but that makes every subsequent `@Test`
  *     reuse the class-scoped ctx — breaking the per-method `Region` invariant that every other
  *     test in this suite checks. Belongs in a dedicated suite.
  *   - Separate `ClassLevelIrCaches` per class (vs. shared `SharedResources` across classes).
  *     Observable only by cross-class state, which would couple this suite to a second suite's
  *     execution order.
  *   - `ExecutionTimer.finished` is private with no accessor, so we can't assert
  *     `OwnedExecuteContext.close` called `timer.finish()` on scope-end.
  *   - Region invalidation on scope-end. The design says JUnit closing the stored
  *     `OwnedExecuteContext` "releases the Region", but `OwnedExecuteContext.close` only calls
  *     `ctx.close()` which closes `tempFileManager` and `taskContext` — not the Region. A
  *     `previousCtx.r.isValid() == false` assertion in `@BeforeEach` of the next method would fail
  *     on the current code. Treating it as a probable code bug rather than asserting a property
  *     that doesn't hold.
  *   - `RVD.CheckRvdKeyOrderingForTesting` side-effect in `shared(...)`. Testable via a single
  *     `assertTrue(RVD.CheckRvdKeyOrderingForTesting)` but belongs elsewhere — it's a one-time
  *     flag, not a lifetime.
  *   - The `@AfterAll` safety check that no test stopped the `SparkContext`. Exercising it requires
  *     stopping the SparkContext mid-suite, which would break every following suite. Not testable
  *     without process-level isolation.
  */
class HailExtensionSuite {
  private var beforeEachCtx: ExecuteContext = _

  private val ctxByMethod: mutable.Map[String, ExecuteContext] = mutable.HashMap.empty
  private val regionByMethod: mutable.Map[String, Region] = mutable.HashMap.empty
  private val timerByMethod: mutable.Map[String, AnyRef] = mutable.HashMap.empty
  private val tempFileMgrByMethod: mutable.Map[String, AnyRef] = mutable.HashMap.empty
  private val irMetadataByMethod: mutable.Map[String, AnyRef] = mutable.HashMap.empty

  private var sharedBackend: Backend = _
  private var sharedFS: FS = _
  private var sharedPool: RegionPool = _
  private var sharedBmCache: AnyRef = _
  private var sharedCompileCache: AnyRef = _
  private var sharedPersistedIrCache: AnyRef = _
  private var sharedPersistedCoercerCache: AnyRef = _

  @BeforeEach
  def captureAndAssertInvariants(ctx: ExecuteContext, info: TestInfo): Unit = {
    val methodName = info.getTestMethod.get().getName
    beforeEachCtx = ctx

    ctxByMethod.get(methodName) match {
      case None =>
        assertTrue(
          !ctxByMethod.values.exists(_ eq ctx),
          s"$methodName received an ExecuteContext already used by another method",
        )
        assertTrue(
          !regionByMethod.values.exists(_ eq ctx.r),
          s"$methodName received a Region already used by another method",
        )
        assertTrue(
          !timerByMethod.values.exists(_ eq ctx.timer),
          s"$methodName received an ExecutionTimer already used by another method",
        )
        assertTrue(
          !tempFileMgrByMethod.values.exists(_ eq ctx.tempFileManager),
          s"$methodName received a tempFileManager already used by another method",
        )
        assertTrue(
          !irMetadataByMethod.values.exists(_ eq ctx.irMetadata),
          s"$methodName received an IrMetadata already used by another method",
        )
        ctxByMethod(methodName) = ctx
        regionByMethod(methodName) = ctx.r
        timerByMethod(methodName) = ctx.timer
        tempFileMgrByMethod(methodName) = ctx.tempFileManager
        irMetadataByMethod(methodName) = ctx.irMetadata

      case Some(priorCtx) =>
        assertSame(priorCtx, ctx, s"$methodName invocations should share one ExecuteContext")
        assertSame(
          regionByMethod(methodName),
          ctx.r,
          s"$methodName invocations should share one Region",
        )
        assertSame(
          timerByMethod(methodName),
          ctx.timer,
          s"$methodName invocations should share one ExecutionTimer",
        )
        assertSame(
          tempFileMgrByMethod(methodName),
          ctx.tempFileManager,
          s"$methodName invocations should share one tempFileManager",
        )
        assertSame(
          irMetadataByMethod(methodName),
          ctx.irMetadata,
          s"$methodName invocations should share one IrMetadata",
        )
    }

    if (sharedBackend == null) {
      sharedBackend = ctx.backend
      sharedFS = ctx.fs
      sharedPool = ctx.r.pool
      sharedBmCache = ctx.BlockMatrixCache
      sharedCompileCache = ctx.CompileCache
      sharedPersistedIrCache = ctx.PersistedIrCache
      sharedPersistedCoercerCache = ctx.PersistedCoercerCache
    } else {
      assertSame(sharedBackend, ctx.backend, "Backend should be shared across tests")
      assertSame(sharedFS, ctx.fs, "FS should be shared across tests")
      assertSame(sharedPool, ctx.r.pool, "RegionPool should be shared across tests")
      assertSame(sharedBmCache, ctx.BlockMatrixCache, "BlockMatrixCache should be class-scoped")
      assertSame(sharedCompileCache, ctx.CompileCache, "CompileCache should be class-scoped")
      assertSame(
        sharedPersistedIrCache,
        ctx.PersistedIrCache,
        "PersistedIrCache should be class-scoped",
      )
      assertSame(
        sharedPersistedCoercerCache,
        ctx.PersistedCoercerCache,
        "PersistedCoercerCache should be class-scoped",
      )
    }

    assertTrue(ctx.r.isValid(), "ExecuteContext's Region should be live during the test")
  }

  @Test def beforeEachAndTestShareCtx(ctx: ExecuteContext): Unit =
    assertSame(beforeEachCtx, ctx)

  @Test def twoInjectionsSameMethodResolveToSameCtx(a: ExecuteContext, b: ExecuteContext): Unit =
    assertSame(a, b)

  @Test def fsParamIsCtxFs(fs: FS, ctx: ExecuteContext): Unit = {
    assertSame(fs, ctx.fs)
    assertSame(beforeEachCtx, ctx)
  }

  @Test def backendParamIsCtxBackend(backend: Backend, ctx: ExecuteContext): Unit =
    assertSame(backend, ctx.backend)

  @Test def poolParamIsCtxPool(pool: RegionPool, ctx: ExecuteContext): Unit =
    assertSame(pool, ctx.r.pool)

  @Test def anotherMethodGetsItsOwnCtx(ctx: ExecuteContext): Unit =
    assertTrue(ctx.r.isValid())

  @Test def yetAnotherMethodGetsItsOwnCtx(ctx: ExecuteContext): Unit =
    assertTrue(ctx.r.isValid())

  def factoryAndTestShareCtx(implicit ctx: ExecuteContext): Seq[Box[ExecuteContext]] =
    Seq.fill(3)(Box(ctx))

  @ParameterizedTest
  def factoryAndTestShareCtx(fromFactory: Box[ExecuteContext], ctx: ExecuteContext): Unit = {
    assertSame(fromFactory.value, ctx, "@ParameterizedTest ctx should equal the factory's ctx")
    assertSame(beforeEachCtx, ctx, "@BeforeEach ctx should equal the factory's ctx")
  }

  def sameRegionAcrossInvocations(implicit ctx: ExecuteContext): Seq[Box[Region]] =
    Seq.fill(3)(Box(ctx.r))

  @ParameterizedTest
  def sameRegionAcrossInvocations(fromFactory: Box[Region], ctx: ExecuteContext): Unit =
    assertSame(fromFactory.value, ctx.r)
}

/** Test-suite-local wrapper used to hide `HailExtension`-provided types (e.g. `ExecuteContext`,
  * `Region`) inside factory-produced arguments. Without the wrapper, both JUnit's built-in
  * `ParameterizedTestMethodParameterResolver` and `HailExtension` claim those parameter slots and
  * the invocation fails with "multiple competing ParameterResolvers". Normal suites never produce
  * such types from a factory — only this suite does, in order to verify that the factory and the
  * test body observe the same Hail state.
  */
final case class Box[T](value: T)
