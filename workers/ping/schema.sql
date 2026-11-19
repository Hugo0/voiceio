-- One row per ISO week and combination of the five reported values; n is how
-- many installs sent that combination that week. Nothing else is stored: no
-- IP, no headers, no time finer than the week.
CREATE TABLE IF NOT EXISTS counts (
  week    TEXT NOT NULL,              -- ISO week of the request, e.g. "2026-W47"
  v       TEXT NOT NULL,              -- voiceio version
  distro  TEXT NOT NULL,              -- "<ID> <VERSION_ID>" from /etc/os-release
  desktop TEXT NOT NULL,
  session TEXT NOT NULL,              -- wayland, x11, ...
  decoder TEXT NOT NULL,              -- parakeet, whisper, ...
  n       INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (week, v, distro, desktop, session, decoder)
);
