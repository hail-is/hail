package is.hail

import is.hail.backend.ExecutionCache
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.{agg, Optimize}
import is.hail.io.fs.RequesterPaysConfig
import is.hail.types.encoded.EType
import is.hail.utils._

import scala.collection.mutable

import org.json4s.JsonAST.{JArray, JObject, JString}

object HailFeatureFlags {
  val defaults = Map[String, (String, String)](
    // Must match _flags_env_vars_and_defaults in hail/backend/backend.py
    //
    // The default values and envvars here are only used in the Scala tests. In all other
    // conditions, Python initializes the flags, see HailContext._initialize_flags in context.py.
    ("distributed_scan_comb_op", ("HAIL_DEV_DISTRIBUTED_SCAN_COMB_OP" -> null)),
    ("grouped_aggregate_buffer_size", ("HAIL_GROUPED_AGGREGATE_BUFFER_SIZE" -> "50")),
    ("index_branching_factor", "HAIL_INDEX_BRANCHING_FACTOR" -> null),
    ("jvm_bytecode_dump", ("HAIL_DEV_JVM_BYTECODE_DUMP" -> null)),
    ("lower", ("HAIL_DEV_LOWER" -> null)),
    ("lower_bm", ("HAIL_DEV_LOWER_BM" -> null)),
    ("lower_only", ("HAIL_DEV_LOWER_ONLY" -> null)),
    ("max_leader_scans", ("HAIL_DEV_MAX_LEADER_SCANS" -> "1000")),
    ("method_split_ir_limit", ("HAIL_DEV_METHOD_SPLIT_LIMIT" -> "16")),
    ("no_ir_logging", ("HAIL_DEV_NO_IR_LOG" -> null)),
    ("no_whole_stage_codegen", ("HAIL_DEV_NO_WHOLE_STAGE_CODEGEN" -> null)),
    ("print_inputs_on_worker", ("HAIL_DEV_PRINT_INPUTS_ON_WORKER" -> null)),
    ("print_ir_on_worker", ("HAIL_DEV_PRINT_IR_ON_WORKER" -> null)),
    ("profile", "HAIL_PROFILE" -> null),
    ("rng_nonce", "HAIL_RNG_NONCE" -> "0x0"),
    ("shuffle_cutoff_to_local_sort", ("HAIL_SHUFFLE_CUTOFF" -> "512000000")), // This is in bytes
    ("shuffle_max_branch_factor", ("HAIL_SHUFFLE_MAX_BRANCH" -> "64")),
    ("use_new_shuffle", ("HAIL_USE_NEW_SHUFFLE" -> null)),
    ("use_ssa_logs", "HAIL_USE_SSA_LOGS" -> "1"),
    ("write_ir_files", ("HAIL_WRITE_IR_FILES" -> null)),
    (agg.Flags.BranchingFactor, "HAIL_BRANCHING_FACTOR" -> null),
    (EType.Flags.UseUnstableEncodings, EType.Flags.UseUnstableEncodingsVar -> null),
    (ExecutionCache.Flags.Cachedir, "HAIL_CACHE_DIR" -> null),
    (ExecutionCache.Flags.UseFastRestarts, "HAIL_USE_FAST_RESTARTS" -> null),
    (RequesterPaysConfig.Flags.RequesterPaysBuckets, "HAIL_GCS_REQUESTER_PAYS_BUCKETS" -> null),
    (RequesterPaysConfig.Flags.RequesterPaysProject, "HAIL_GCS_REQUESTER_PAYS_PROJECT" -> null),
    (
      SparkBackend.Flags.MaxStageParallelism,
      "HAIL_SPARK_MAX_STAGE_PARALLELISM" -> Integer.MAX_VALUE.toString,
    ),
    (Optimize.Flags.MaxOptimizerIterations, "HAIL_OPTIMIZER_ITERATIONS" -> null),
    (Optimize.Flags.Optimize, "HAIL_QUERY_OPTIMIZE" -> "1"),
  )

  def fromEnv(m: Map[String, String] = sys.env): HailFeatureFlags =
    new HailFeatureFlags(
      mutable.Map(
        HailFeatureFlags.defaults.map {
          case (flagName, (_, default)) => (flagName, m.getOrElse(flagName, default))
        }.toFastSeq: _*
      )
    )
}

class HailFeatureFlags private (
  private[this] val flags: mutable.Map[String, String]
) extends Serializable {
  val available: java.util.ArrayList[String] =
    new java.util.ArrayList[String](java.util.Arrays.asList[String](flags.keys.toSeq: _*))

  def set(flag: String, value: String): Unit = {
    assert(exists(flag))
    flags.update(flag, value)
  }

  def +(feature: (String, String)): HailFeatureFlags =
    new HailFeatureFlags(flags.clone() += feature._1 -> feature._2)

  def define(feature: String): HailFeatureFlags =
    new HailFeatureFlags(flags.clone() += feature -> "1")

  def -(feature: String): HailFeatureFlags =
    new HailFeatureFlags(flags.clone() -= feature)

  def get(flag: String): String = flags(flag)

  def lookup(flag: String): Option[String] =
    flags.get(flag).flatMap(Option(_))

  def isDefined(flag: String): Boolean =
    lookup(flag).isDefined

  def exists(flag: String): Boolean =
    flags.contains(flag)

  def toJSONEnv: JArray =
    JArray(flags.filter { case (_, v) =>
      v != null
    }.map { case (name, v) =>
      JObject(
        "name" -> JString(HailFeatureFlags.defaults(name)._1),
        "value" -> JString(v),
      )
    }.toList)
}
