# Source fidelity and source factuality

Checked on 2026-11-02 KST. These findings concern the original Korean sources,
not a change introduced by translation. The frozen corpus and expert references
were not rewritten during evaluation. A high reference score or exact quotation
does not resolve these issues.

- `nella-fantasia-ku-013`: the source spells the composer's first name Enrico.
  Sarah Brightman's own album page credits **Ennio Morricone** for Nella Fantasia.
  This supports a source-name correction in a future curated corpus revision.
  [Official Eden album page](https://sarahbrightman.com/discography/eden/).
- `in-flowery-clouds-ku-012`: the source first discourages an extremely slow
  reading of Larghetto, then describes it as very slow. The Open University's
  teaching table distinguishes Larghetto as slower music that is **less slow
  than Largo**. This supports the judge's flag that the source wording needs
  clarification; it does not establish a unique BPM for this song.
  [Open University performance directions](https://www.open.edu/openlearn/history-the-arts/music/an-introduction-music-theory/content-section-8.1).
- `la-capinera-ku-013`: Claude/Codex flagged treating a 2–3-times fermata length
  as a general principle. Preserve the distinction between this teacher's local
  performance recommendation and a universal notation rule. This note records
  a review issue, not a newly expert-approved replacement answer.
- `die-forelle-ku-003`: the source's historical “first composer” claim was also
  flagged in the catalog review. It needs musicological source checking before
  a definitive correction; no replacement historical claim was invented here.

The first catalog review and all translation revisions are in
`translation-catalog-review.json` and the versioned review directories.
The runtime source quotations and reviewed translations can retain these
original problems. Report factuality separately from semantic fidelity and
keep their contribution to whole-service quality visible.
