package is.hail

import is.hail.backend.ExecutionCache
import is.hail.utils._

import org.json4s.JsonAST.{JArray, JObject, JString}

import scala.collection.mutable

object HailFeatureFlags {
  val defaults = Map[String, (String, String)](
    // Must match __flags_env_vars_and_defaults in hail/backend/backend.py
    //
    // The default values and envvars here are only used in the Scala tests. In all other
    // conditions, Python initializes the flags, see HailContext._initialize_flags in context.py.
    ("no_whole_stage_codegen", ("HAIL_DEV_NO_WHOLE_STAGE_CODEGEN" -> null)),
    ("no_ir_logging", ("HAIL_DEV_NO_IR_LOG" -> null)),
    ("lower", ("HAIL_DEV_LOWER" -> null)),
    ("lower_only", ("HAIL_DEV_LOWER_ONLY" -> null)),
    ("lower_bm", ("HAIL_DEV_LOWER_BM" -> null)),
    ("print_ir_on_worker", ("HAIL_DEV_PRINT_IR_ON_WORKER" -> null)),
    ("print_inputs_on_worker", ("HAIL_DEV_PRINT_INPUTS_ON_WORKER" -> null)),
    ("max_leader_scans", ("HAIL_DEV_MAX_LEADER_SCANS" -> "1000")),
    ("distributed_scan_comb_op", ("HAIL_DEV_DISTRIBUTED_SCAN_COMB_OP" -> null)),
    ("jvm_bytecode_dump", ("HAIL_DEV_JVM_BYTECODE_DUMP" -> null)),
    ("write_ir_files", ("HAIL_WRITE_IR_FILES" -> null)),
    ("method_split_ir_limit", ("HAIL_DEV_METHOD_SPLIT_LIMIT" -> "16")),
    ("use_new_shuffle", ("HAIL_USE_NEW_SHUFFLE" -> null)),
    ("shuffle_max_branch_factor", ("HAIL_SHUFFLE_MAX_BRANCH" -> "64")),
    ("shuffle_cutoff_to_local_sort", ("HAIL_SHUFFLE_CUTOFF" -> "512000000")), // This is in bytes
    ("grouped_aggregate_buffer_size", ("HAIL_GROUPED_AGGREGATE_BUFFER_SIZE" -> "50")),
    ("use_ssa_logs", "HAIL_USE_SSA_LOGS" -> "1"),
    ("gcs_requester_pays_project", "HAIL_GCS_REQUESTER_PAYS_PROJECT" -> null),
    ("gcs_requester_pays_buckets", "HAIL_GCS_REQUESTER_PAYS_BUCKETS" -> null),
    ("index_branching_factor", "HAIL_INDEX_BRANCHING_FACTOR" -> null),
    ("rng_nonce", "HAIL_RNG_NONCE" -> "0x0"),
    ("profile", "HAIL_PROFILE" -> null),
    (ExecutionCache.Flags.UseFastRestarts, "HAIL_USE_FAST_RESTARTS" -> null),
    (ExecutionCache.Flags.Cachedir, "HAIL_CACHE_DIR" -> null),
  )

  def fromEnv(): HailFeatureFlags =
    new HailFeatureFlags(
      mutable.Map(
        HailFeatureFlags.defaults.mapValues { case (env, default) =>
          sys.env.getOrElse(env, default)
        }.toFastSeq: _*
      )
    )

  def fromMap(m: Map[String, String]): HailFeatureFlags =
    new HailFeatureFlags(
      mutable.Map(
        HailFeatureFlags.defaults.map {
          case (flagName, (_, default)) => (flagName, m.getOrElse(flagName, default))
        }.toFastSeq: _*
      )
    )
}

class HailFeatureFlags private (
  val flags: mutable.Map[String, String]
) extends Serializable {
  val available: java.util.ArrayList[String] =
    new java.util.ArrayList[String](java.util.Arrays.asList[String](flags.keys.toSeq: _*))

  def set(flag: String, value: String): Unit = {
    assert(exists(flag))
    flags.update(flag, value)
  }

  def get(flag: String): String = flags(flag)

  def exists(flag: String): Boolean = flags.contains(flag)

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
