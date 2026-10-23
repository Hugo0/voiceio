# Learning your voice

Most dictation errors are proper nouns and jargon: `PostgreSQL` heard as `post grass`, a colleague's name spelled three ways. There are three tools for that, from cheapest to most powerful.

**1. Vocabulary.** Put your names and terms in `~/.config/voiceio/vocabulary.txt` (or use `voiceio vocab`). They are passed to Whisper as *hotwords* on every decode, ranked by how often you use them. One honest limit: Whisper's hotword channel is capped at 223 tokens, and a proper noun costs 5 to 7 tokens, so only about **35 terms** steer each decode. The rest are kept and ranked, but they do nothing until they rank high enough. This ceiling comes from Whisper, not from voiceio. With Parakeet as the decoder (`voiceio models use parakeet-tdt-0.6b-v2`) there is no cap: every proper noun in your vocabulary is a hotword. [CONTRIBUTING.md](../CONTRIBUTING.md#model-choice-parakeet-where-it-can-be-biased-whisper-otherwise) has the measurements.

**2. Corrections.** Find/replace rules for what still comes out wrong (`voiceio corrections add WRONG RIGHT`, or say "correct that" right after a mistake and review with `voiceio corrections flagged`). `voiceio vocab suggest` lists unusual words from your history — offline — so you can add the real names to your vocabulary and turn the mishearings into corrections.

**3. Fine-tuning with `voiceio learn`.** This is how you get past the 35-term ceiling: teach the model your vocabulary directly, from your own retained audio.

```bash
voiceio learn import DIR --source phone   # optional: add your own recordings from elsewhere
voiceio learn label      # a stronger teacher model (large-v3-turbo) transcribes your clips
voiceio learn review     # fix the teacher where it matters: your names and terms
voiceio learn review --test   # verify the held-out clips, so eval scores against the truth: it asks only
                              #   where the teacher and your live decode disagree, quickest clips first
voiceio learn augment    # synthesize extra examples of rare terms with local TTS
voiceio learn build      # freeze a versioned train/dev/test dataset (the test split never trains)
voiceio learn eval       # score the current model on your held-out clips: WER and term recall
                         #   --streaming replays them through live dictation and times stop→text
voiceio learn train      # LoRA fine-tune (needs the `train` extra; decoder-only on CPU, --full on a GPU)
voiceio learn promote <run>   # dictate with the fine-tune; `voiceio learn rollback` undoes it
voiceio learn maintain   # later: label what's new, and once 50+ clips piled up ([learn] min_new), retrain and compare
voiceio learn schedule on   # …or let it run `maintain` on a schedule, and notify you when a model wins
                            #   --at 02:00 --every day: when (default weekly, Sundays at 03:30)
                            #   --auto-promote: switch to a winner by itself; `learn rollback` undoes it
voiceio learn stop          # stop a run in progress (it goes again next time)
```

Scheduled runs never compete with you: they work only while you're away from the computer. A run starts once there has been no keyboard or mouse input for `[learn] away_minutes` (10), or the screen is locked, on mains power, with the 1-minute load per CPU at most `[learn] max_load` (0.25). While it works it checks every two seconds, and the moment you touch the keyboard or mouse (or unplug the charger) it stops, within about two seconds. It carries on where it was the next time you're away. Labels are saved per clip, training checkpoints about once a minute and keeps the encoder outputs it computed on disk, and finished evals are reused, so an interruption costs at most a minute of work. Each wait lasts at most `wait_hours` (3); after that the next scheduled run takes over, so a laptop that was off at 03:30 catches up during the day only while you're away, or else the next night. When a run starts you get a notification with **Stop for now** and **Weekly instead**.

How "away" is read, best source first: GNOME's idle monitor (`org.gnome.Mutter.IdleMonitor`, the time since your last input) or KDE's (`org.freedesktop.ScreenSaver`), a locked or blanked screen, and logind's idle hint on other desktops. An app that keeps the session awake, such as a video or a call, counts as you being there. When none of these can be read, the computer counts as in use and nothing runs. `journalctl --user -u voiceio-learn` shows what it waited for and when it paused.

What a scheduled run trains comes from `[learn]` too: `base` (default `small`; any Whisper size, `.en` included, or a Hugging Face checkpoint), `teacher` (default `large-v3-turbo`), `epochs` (4) and `min_new` (50); `maintain`'s flags override them. To fine-tune the English-only medium model: `voiceio config set learn.base medium.en` (on CPU, medium trains and dictates roughly three times slower than small).

To keep a second machine (say, the server your phone dictates through) on the same model, set `[learn] on_promote = "scripts/push_model.sh HOST"`: after every switch it copies the model there and flips a `current` link the server points at.

Labeling, training and offline eval also run in an idle-priority cgroup. The scheduled unit adds `CPUSchedulingPolicy=idle`, idle disk priority, `MemoryHigh=25%` (past it the kernel reclaims from training, not from your desktop) and `OOMScoreAdjust=500`. Labeling and training use half the logical CPUs (`[learn] threads`, 0 = half). Everything stays on your machine unless you choose `train --remote`. An interrupted `voiceio learn train` resumes too: run it again with the same `--base` on the same dataset.

`voiceio learn status` shows where you are, each model's scores, and suggests `promote` only when a fine-tune wins on everything. `promote` itself refuses a model that hasn't, and reports how many clips got better or worse with a confidence interval — a few dozen test clips can move an average by luck.

**Is it actually helping?** Once some clips are labeled, measure instead of guessing:

```bash
voiceio learn eval --pipeline   # what each text stage (numbers, commands, corrections, cloud
                                # postcorrect…) did to your past dictations, and its commonest edits
voiceio corrections audit       # rules your own audio contradicts, and ones that look risky
voiceio learn eval --streaming  # accuracy *and* stop→text, replaying clips through live dictation
```

On the author's data these found a correction that kept turning "composable" into "comparable", a number rule writing "no one" as "no 1", and a cloud pass that never fixed a single vocabulary term.
