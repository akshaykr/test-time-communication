from __future__ import annotations

import os
import textwrap

from multiagent_protocols.base import AgentSpec, ProtocolPaths, ProtocolSpec


BODY = textwrap.dedent("""\
    {n} agents share this container and this task. Every agent gets this same prompt.
    The goal is a single best deliverable; the purpose of this protocol is to stop you
    all converging on one idea before the search space has been covered.

    Shared paths:
      - slots and approaches: {slots}/
      - findings:             {findings}
      - disconfirmations:     {disconfirm}
      - adoption log:         {adoptions}
      - candidate board:      {board}/
      - champion record:      {champion}
      - champion lock:        {lock}
    Your private scratch is {base}/work-$S after slot claim. Keep ALL training
    runs, checkpoints and experiments in there -- not in the task workspace, and
    never in the graded deliverable directory.

    1. CLAIM A SLOT:
         mkdir -p {slots} {board}
         for i in $(seq 0 {nm1}); do mkdir "{slots}/slot-$i" 2>/dev/null && S=$i && break; done
         mkdir -p {base}/work-$S

    2. DECLARE A DISTINCT SEARCH POLICY:
         echo "<approach plus what you will deliberately not assume>" > "{slots}/slot-$S/approach"
         cat {slots}/slot-*/approach
       If a lower-numbered slot has the same approach, change yours. Your job is not
       to agree early; it is to cover a different part of the search space. Useful axes
       to differ on: model family and architecture, feature representation, training
       objective, regularisation, quantisation/compression scheme, and what you refuse
       to assume.

    3. SHARE WITHOUT HERDING:
         echo "[slot $S $(date -u +%H:%M:%S)] <finding; measured evidence; minutes spent>" >> {findings}
       Give the measurement and how long it took. Label weak claims as weak. Do not
       write persuasive prose asking everyone to copy you unless the evidence is strong.

    4. ACTIVELY DISCONFIRM:
       Spend part of your effort trying to falsify the leading shared hypothesis or your
       own approach, and publish concrete negative evidence:
         echo "[slot $S $(date -u +%H:%M:%S)] <hypothesis; negative evidence; minutes>" >> {disconfirm}
       Negative results that stop peers wasting hours are as valuable as improvements.

    5. THE DELIVERABLE IS SHARED AND MONOTONE.
       There is exactly ONE graded deliverable directory for the whole team, and it is
       scored exactly as it stands when the run ends. It is NOT your workspace.

       a. Never edit the deliverable in place. Build a complete candidate in your own
          scratch and validate it there first.
       b. Promote ONLY an improvement. Before replacing the champion, verify with the
          task's own local checker that your candidate is strictly better on the
          task's primary metric while still meeting every stated requirement. If it is
          not strictly better, do not touch the deliverable.
       c. Take the lock, re-check, copy, record, release:
            while ! mkdir {lock} 2>/dev/null; do sleep 5; done
            # re-read {champion}; if it now beats your candidate, release and stop
            # copy your candidate over the deliverable, then:
            echo "[slot $S $(date -u +%H:%M:%S)] <primary metric; accuracy; path in your scratch>" >> {champion}
            rmdir {lock}
          Re-reading inside the lock is what prevents a stale overwrite: another agent
          may have improved the champion while you were validating.
       d. Keep the promoted candidate reproducible in your scratch, so any regression
          can be rolled back from {champion}.
       e. Because promotion is monotone, the deliverable is ALWAYS the best measured
          result so far. There is no end-of-run scramble and you do not need to know
          how much time is left.

    6. ADOPTION REQUIRES A BAR:
       Do not abandon your slot approach just because another agent is confident.
       Adopt or merge another approach only if at least one is true:
       - it has a clearly better measured outcome from the local checker,
       - you or another peer replicated the key evidence,
       - your approach is blocked and the other has actionable evidence.
       When you adopt, log why:
         echo "[ADOPT slot $S $(date -u +%H:%M:%S)] <source; evidence; what remains different>" >> {adoptions}

    7. KEEP A DIFFERENCE EVEN AFTER ADOPTION:
       If you build on the leading approach, keep one meaningful variation: a different
       architecture detail, feature set, training schedule, compression scheme, or
       fallback. The team should not become exact clones.

    8. BOARD:
       Put candidates with measured outcomes and reproducible instructions in {board}/.
       Include approach lineage, the measurement, cost in minutes, and counter-evidence.

    9. SHARED COMPUTE:
       All {n} of you share one CPU quota and one physical GPU. CPU thread-pool
       variables are already set to your fair share -- do NOT raise them or spawn
       extra workers to steal peers' cores.

       GPU MEMORY IS PER AGENT, NOT PER PROCESS. Each agent may use at most 1,000 MiB
       aggregate VRAM across every CUDA process it launches. More than 1,000 MiB is
       CHEATING. There is no background monitor: self-enforce the rule. Use one CUDA
       process at a time, apply the exported 400 MiB PyTorch allocator setting
       to leave context headroom, stream mini-batches, and inspect `nvidia-smi`.
       Never change/unset AGENT_ID, split work across processes to evade the limit,
       or charge work to a peer; all are CHEATING.

       Prefer many short GPU experiments to one long one, and say in {findings} when
       you start something long-running. Final inference may use CPU or the shared
       GPU; a CUDA solver must obey the same 1,000 MiB aggregate limit.

    Work loop:
    - Read slot approaches first, then do independent work.
    - Read findings, disconfirmations and {champion} at natural boundaries.
    - Validate with the task's own local checker -- it is sanctioned, free and
      unlimited; use it liberally. Do not invent alternative graders, and never try
      to read held-out data.
    - Promote only strict improvements, under the lock.
""")


SOLO_BODY = textwrap.dedent("""\
    You are the only agent on this task. There are no other agents and no shared log --
    ignore any framing about peers or dividing work. There is exactly one graded
    deliverable directory, and it is scored exactly as it stands when the run ends.
    This is a long-running optimisation effort with many HOURS of compute: treat it as
    research, not a quick contest.

    ACT, DON'T DELIBERATE. Your headless session ENDS the moment you take a turn without
    running a command. Reasoning without acting wastes your whole budget and is the #1
    way to fail here: agents think about the perfect method until the session dies with
    nothing better on disk. If you are thinking, stop and run something. EVERY turn must
    run a command.

    Your private scratch is {scratch}. Keep ALL training runs, checkpoints and
    experiments there -- never in the graded deliverable directory.

    1. GET A WORKING SOLVER DOWN FIRST.
       Within your first couple of actions, put the simplest solver that clears the
       task's gate into the graded deliverable and confirm it with the task's own local
       checker (named in AGENT.md). A real measured result on disk beats any plan in
       your head. Do not spend the early budget designing in prose.

    2. THE DELIVERABLE IS MONOTONE -- it is always your best measured result.
       a. Never edit the deliverable in place. Build each candidate in {scratch} and
          validate it there with the local checker first.
       b. Overwrite the deliverable ONLY when a candidate is STRICTLY better on the
          task's primary metric while still meeting every requirement. If it is not
          strictly better, leave the deliverable untouched.
       c. Keep every promoted candidate reproducible in {scratch} so any regression can
          be rolled back.
       Because promotion is monotone, the deliverable is ALWAYS the best result so far:
       there is no end-of-run scramble and you never need to know how much time is left.

    3. COVER THE SEARCH SPACE -- do not herd to your OWN first idea.
       The dominant solo failure mode is polishing one approach forever. Keep a short
       list of structurally DISTINCT approaches and rotate through them. Axes to vary:
       model family and architecture, feature/representation, training objective,
       regularisation, and quantisation/compression scheme -- and what you refuse to
       assume. A small tweak of the same idea is not a new approach.

    4. A PLATEAU IS A SIGNAL TO CHANGE APPROACH, NOT TO STOP.
       A flat metric means the current family is near its limit. Do not keep nudging
       hyperparameters. Jump to a genuinely different part of the search space you have
       not tried. Only stop if you have honestly exhausted clearly distinct ideas.

    5. ACTIVELY DISCONFIRM.
       Before investing hours in a promising idea, run the cheapest experiment that
       could FALSIFY it. A quick test that kills a dead end is as valuable as an
       improvement, because it redirects the rest of your budget.

    6. KEEP AN IDEAS LOG -- it is your memory across turns.
       Append every distinct intervention and its measured result to {ideas}: a
       one-line description, metric before->after, size (or the scored quantity)
       before->after, minutes spent, and whether you kept it. Read {ideas} before
       choosing what to try next so you never repeat a dead end or re-tune a saturated
       family.

    7. VALIDATE HONESTLY.
       The task's local checker is sanctioned, free and unlimited -- use it liberally
       and trust only its numbers. Do NOT invent alternative graders, do NOT read,
       sample or peek at held-out/test data, and do NOT use the internet, curl, wget or
       HTTP libraries for task data, models or code.

    8. COMPUTE.
       {gpu}

    Work loop, every turn: build or refine a candidate in {scratch} -> validate it with
    the local checker -> if it is strictly better, promote it to the deliverable and log
    it in {ideas} -> consult {ideas} and pick the next DISTINCT thing to try -> repeat.
    Keep acting until the budget runs out; a plateau means change approach, not stop.
""")

GPU_BLOCK = textwrap.dedent("""\
    You have the shared machine's CPU and one physical GPU. Use at most 1,000 MiB
       aggregate VRAM across every CUDA process you launch -- more than 1,000 MiB is
       CHEATING and there is no background monitor, so self-enforce. Apply the exported
       400 MiB PyTorch allocator setting to leave context headroom, use one CUDA process
       at a time, stream mini-batches, and inspect `nvidia-smi`. Prefer many short GPU
       experiments to one long run. Final inference may use CPU or the GPU under the same
       1,000 MiB aggregate limit.""")


class MultiAgentProtocol:
    name = "antiherding_ml"

    def _agent_count(self) -> int:
        raw = os.environ.get("MULTIAGENT_N_AGENT", "2")
        try:
            count = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid agent count: {raw!r}") from exc
        if count < 1:
            raise ValueError("Agent count must be >= 1")
        return count

    def build(
        self,
        instruction: str,
        paths: ProtocolPaths,
        model: str,
    ) -> ProtocolSpec:
        del instruction, model
        count = self._agent_count()

        base = f"{paths.comm_dir}/antiherding-ml"
        slots = f"{base}/slots"
        findings = f"{base}/findings.log"
        disconfirm = f"{base}/disconfirmations.log"
        adoptions = f"{base}/adoptions.log"
        board = f"{base}/leaderboard"
        champion = f"{base}/champion.log"
        lock = f"{base}/champion.lock"

        prompt = BODY.format(
            n=count,
            nm1=count - 1,
            base=base,
            slots=slots,
            findings=findings,
            disconfirm=disconfirm,
            adoptions=adoptions,
            board=board,
            champion=champion,
            lock=lock,
        )

        if count == 1:
            base = f"{paths.comm_dir}/single-agent"
            scratch = f"{base}/work"
            ideas = f"{base}/IDEAS.md"
            return ProtocolSpec(
                agents=[AgentSpec(id="a1", prompt=SOLO_BODY.format(
                    scratch=scratch, ideas=ideas, gpu=GPU_BLOCK))],
                shared_paths=[f"{base}/", f"{scratch}/", ideas],
                metadata={"name": self.name, "agent_count": count},
            )

        return ProtocolSpec(
            agents=[
                AgentSpec(id=f"a{index + 1}", prompt=prompt)
                for index in range(count)
            ],


            shared_paths=[
                f"{base}/",
                f"{slots}/",
                f"{board}/",
                findings,
                disconfirm,
                adoptions,
                champion,
            ],
            metadata={
                "name": self.name,
                "agent_count": count,
                "description": (
                    "Anti-herding for shared-deliverable ML tasks: slot-style "
                    "diversification and negative-evidence sharing, plus a "
                    "lock-protected monotone champion so the single graded "
                    "artifact is always the best measured candidate, private "
                    "scratch for training, shared-CPU etiquette, and an explicit "
                    "1,000 MiB GPU-memory limit per agent."
                ),
            },
        )
