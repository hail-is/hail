package is.hail.io.reference

import is.hail.utils.fatal
import is.hail.variant.{ReferenceGenome, ReferenceGenomeEnv}

/** A session's reference genome attachments, keyed by the reference genomes they are attached to:
  * at most one liftover per source and destination, and at most one reference sequence per
  * reference genome. Keys compare as reference genomes do, so a reference genome equal to the one
  * an attachment holds finds it, and one redefined under the same name does not.
  *
  * Two of these are equal if they hold the same attachment objects under equal keys.
  */
final case class RgAttachments(
  liftovers: Map[(ReferenceGenome, ReferenceGenome), Liftover],
  sequences: Map[ReferenceGenome, ReferenceSequence],
) {

  /** The liftover from `source` to `dest`. Fails if there is none. */
  def liftover(source: ReferenceGenome, dest: ReferenceGenome): Liftover =
    liftovers.getOrElse(
      (source, dest),
      fatal(s"No liftover from reference genome '${source.name}' to '${dest.name}' has been added."),
    )

  /** The reference sequence of `rg`. Fails if there is none. */
  def sequence(rg: ReferenceGenome): ReferenceSequence =
    sequences.getOrElse(
      rg,
      fatal(s"FASTA file has not been loaded for reference genome '${rg.name}'."),
    )

  /** These attachments, plus `lo`. Fails if there is already a liftover between its genomes. */
  def withLiftover(lo: Liftover): RgAttachments = {
    if (liftovers.contains(lo.key))
      fatal(
        s"LiftOver already exists for source reference '${lo.source.name}' and destination reference '${lo.dest.name}'."
      )
    copy(liftovers = liftovers + (lo.key -> lo))
  }

  /** These attachments, plus `seq`. Fails if its genome already has a reference sequence. */
  def withSequence(seq: ReferenceSequence): RgAttachments = {
    if (sequences.contains(seq.rg))
      fatal(s"FASTA sequence has already been loaded for reference genome '${seq.rg.name}'.")
    copy(sequences = sequences + (seq.rg -> seq))
  }

  /** These attachments, less the liftover from `source` to `dest`, if there is one. */
  def withoutLiftover(source: ReferenceGenome, dest: ReferenceGenome): RgAttachments =
    copy(liftovers = liftovers - ((source, dest)))

  /** These attachments, less the reference sequence of `rg`. Fails if there is none. */
  def withoutSequence(rg: ReferenceGenome): RgAttachments = {
    if (!sequences.contains(rg))
      fatal(s"Reference genome '${rg.name}' does not have sequence loaded.")
    copy(sequences = sequences - rg)
  }

  /** These attachments, less every one `rg` is part of: its reference sequence, and liftovers from
    * and to it.
    */
  def without(rg: ReferenceGenome): RgAttachments =
    RgAttachments(
      liftovers.filter { case ((source, dest), _) => source != rg && dest != rg },
      sequences - rg,
    )
}

object RgAttachments {
  val empty: RgAttachments = RgAttachments(Map.empty, Map.empty)

  /** `env` less the reference genome named `name`, and `attachments` less every attachment it is
    * part of, so that an identical reference genome added later starts with none. Fails if `name`
    * is a builtin's.
    */
  def removeReference(env: ReferenceGenomeEnv, attachments: RgAttachments, name: String)
    : (ReferenceGenomeEnv, RgAttachments) = {
    val without = env.without(name)
    (without, env.get(name).fold(attachments)(attachments.without))
  }

  /** `attachments`, less the liftover between the genomes named `source` and `dest` in `env`. If
    * either is not in scope, nothing can be attached to it, so `attachments` is returned unchanged.
    */
  def removeLiftover(
    env: ReferenceGenomeEnv,
    attachments: RgAttachments,
    source: String,
    dest: String,
  ): RgAttachments =
    (env.get(source), env.get(dest)) match {
      case (Some(s), Some(d)) => attachments.withoutLiftover(s, d)
      case _ => attachments
    }
}
