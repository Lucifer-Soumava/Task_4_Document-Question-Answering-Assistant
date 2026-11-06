# Sample Q&A Transcripts

The following examples were run against the demonstration documents in `data/sample_corpus/`. The answer sentences are copied from the document text and show the citation format produced by the application.

## Transcript 1 — Supported library question

**Question**

How long may an undergraduate borrow a book?

**Assistant answer**

> Undergraduate students may borrow up to four books at a time for 14 days.

**Citation**

`library_policy.txt | paragraphs 1–3 | chunk 1`

**Supporting source passage (excerpt)**

> Undergraduate students may borrow up to four books at a time for 14 days. Students may renew a book once if no other reader has placed a reservation on it. Reference books, newspapers, and current journal issues must be used inside the library and cannot be borrowed.

## Transcript 2 — Supported IT question

**Question**

Who should I contact if password reset does not work?

**Assistant answer**

> If the reset process does not work, contact the IT Help Desk at helpdesk@gvc.example.edu during working hours.

**Citation**

`campus_it_guide.txt | paragraphs 1–4 | chunk 1`

**Supporting source passage (excerpt)**

> For a forgotten password, use the Password Reset option on the college account portal. If the reset process does not work, contact the IT Help Desk at helpdesk@gvc.example.edu during working hours. Include your student registration number in the support request, but never send your password.

## Transcript 3 — Unsupported question

**Question**

What is the college parking fee?

**Assistant answer**

> Information not found in provided document

**Reason**

Neither sample document contains a supporting passage about a parking fee. The assistant does not invent an answer or attach an unrelated citation.

> These transcripts use the bundled sample corpus. Upload your own permitted documents in the app to test a different document collection.
