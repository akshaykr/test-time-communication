from __future__ import annotations

import os
import textwrap

from multiagent_protocols.base import AgentSpec, ProtocolPaths, ProtocolSpec


# Slot-based protocol with explicit anti-herding rules. It preserves independent
# search unless evidence crosses a high bar for adoption.
BODY = textwrap.dedent("""\
    {n} agents share this container and solve the same task together. Identical prompt
    for all agents. This protocol is designed to prevent premature convergence.

    Shared paths:
      - slots and approaches: {slots}/
      - findings:            {findings}
      - disconfirmations:    {disconfirm}
      - adoption log:        {adoptions}
      - attempts board:      {board}/
    Your private scratch is {base}/work-$S after slot claim.

    1. CLAIM A SLOT:
         mkdir -p {slots} {board}
         for i in $(seq 0 {nm1}); do mkdir "{slots}/slot-$i" 2>/dev/null && S=$i && break; done
         mkdir -p {base}/work-$S

    2. DECLARE A DISTINCT SEARCH POLICY:
         echo "<approach plus what you will deliberately not assume>" > "{slots}/slot-$S/approach"
         cat {slots}/slot-*/approach
       If a lower-numbered slot has the same approach, change yours. Your job is not
       to agree early; your job is to cover a different part of the search space.

    3. SHARE WITHOUT HERDING:
       Append concise findings:
         echo "[slot $S $(date -u +%H:%M:%S)] <tier; finding; evidence; cost>" >> {findings}
       Include evidence and cost. Label weak claims as weak. Do not write persuasive
       prose that asks everyone to copy you unless the evidence is strong.

    4. ACTIVELY DISCONFIRM:
       Spend part of your work trying to falsify the leading shared hypothesis or
       your own approach. Append concrete negative evidence to {disconfirm}:
         echo "[slot $S $(date -u +%H:%M:%S)] <hypothesis tested; negative evidence; cost>" >> {disconfirm}

    5. ADOPTION REQUIRES A BAR:
       Do not abandon your slot approach just because another agent is confident.
       Adopt or merge another approach only if at least one is true:
       - it has a clearly better measured outcome from an allowed check,
       - you or another peer replicated the key evidence,
       - your current approach is blocked and the other approach has actionable evidence,
       - the run is near finalization and the team needs one best candidate.

       When you adopt, log why:
         echo "[ADOPT slot $S $(date -u +%H:%M:%S)] <source; evidence; what remains different>" >> {adoptions}

    6. KEEP A DIFFERENCE EVEN AFTER ADOPTION:
       If you build on the leading approach, keep one meaningful variation: a different
       parameter, subcase, proof path, implementation path, check, or fallback. The team
       should not become exact clones until final packaging.

    7. BOARD:
       Use {board}/ for candidates with measured outcomes or reproducible instructions.
       Include approach lineage, evidence, cost, and known counter-evidence.

    Work loop:
    - Read slot approaches first, then do independent work.
    - Read findings and disconfirmations at natural boundaries.
    - Publish useful evidence, especially negative evidence that prevents wasted work.
    - Only converge when the adoption bar is met.
    - Every turn must run a command: claim, build, test, disconfirm, publish, adopt, or update.

    Do not use hidden verifiers, internet access, curl, wget, HTTP libraries, secrets,
    or raw environment dumps. If validation spends task budget, record the cost.
""")


class MultiAgentProtocol:
    name = "antiherding"

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

        base = f"{paths.comm_dir}/anti-herding"
        slots = f"{base}/slots"
        findings = f"{base}/findings.log"
        disconfirm = f"{base}/disconfirmations.log"
        adoptions = f"{base}/adoptions.log"
        board = f"{base}/leaderboard"

        prompt = BODY.format(
            n=count,
            nm1=count - 1,
            base=base,
            slots=slots,
            findings=findings,
            disconfirm=disconfirm,
            adoptions=adoptions,
            board=board,
        )

        return ProtocolSpec(
            agents=[
                AgentSpec(id=f"a{index + 1}", prompt=prompt)
                for index in range(count)
            ],
            shared_paths=[],
            metadata={
                "name": self.name,
                "agent_count": count,
                "description": (
                    "Slot-style diversification with explicit anti-herding rules. Agents "
                    "preserve distinct approaches, publish negative evidence, and only "
                    "adopt another approach after measured, replicated, blocked, or final "
                    "convergence conditions are met."
                ),
            },
        )
