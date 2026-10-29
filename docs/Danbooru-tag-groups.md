# Tag subgroup snapshot

The tag filter uses the hierarchy at https://danbooru.donmai.us/wiki_pages/tag_groups
(retrieved October 28, 2026) and tag links from the wiki pages listed there.
`Sources/Waifu/Model/DanbooruTagGroups.swift` contains the resulting offline
membership table. No network request is made while displaying tags.

The most deeply nested membership takes precedence when a tag belongs to more
than one list; ties use the index page's order. Unlisted tags appear in Other tags.
The snapshot is a navigation aid: linked wiki lists may include related tags as
well as primary members, and future Danbooru edits are not automatically imported.

The existing chip ordering and selection actions are preserved within each group.
Quick filtering applies to every tag, including selected tags. Clearing the quick
filter restores those tags without changing their selection. Empty branches are
omitted. Branches begin expanded and typing a quick filter expands matches.

The UI shows one level of broad categories. Groups are ordered
alphabetically A–Z before quick filtering. Quick
filtering only removes nonmatching tags and empty groups; it never changes the
relative group order. The existing tag order is preserved inside each group.
