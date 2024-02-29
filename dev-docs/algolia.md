# Algolia

Algolia is a hosted search product that we use for documentation search. The credentials are in the
usual place.

We have three "applications":

1. BH4D9OD16A "DocSearch"
2. NDJ9H4VRX7
3. SWB3TKBY4S "hail_is"

The first two ought to be deleted but I do not know how to do that. The second one is just a
mistake. The first is a legacy application which we cannot use anymore.

Within an application there are one or more indices. Our current active one is called "hail_is". I
think the idea is that you can iterate on a non-live index and then switch your application to point
at a different index.

An index is a collection of records including at least a "url" and a "hierarchy". The records, by
way of the hierarchy, form a tree. Terminal records of the tree have one more property:
"content". The content for us might be all the documentation for a single method or a single doc page.

The hierarchy informs how the search results are displayed. Items are shown with at least their
"lvl0" header.

## Crawlers and DocSearch

These indices are automatically created by a "crawler" called "DocSearch". This is also a hosted
product. The URL for this product is absurdly hard to find. For our current application it is:
https://crawler.algolia.com/admin/crawlers?sort=status&order=ASC&limit=20&appId=SWB3TKBY4S .

Click through to the "hail_is" index and go to the "Editor". Ignore most of the information here and
look at the `helpers.docsearch` method invocation. This is how we explain to DocSearch how to
convert our web pages into records for the index. As an aside: you might think this is silly we
should generate records from our structured documentation not from the HTML we generate. I
agree. Alas, I am writing this and you are reading this.

```
return helpers
  .docsearch({
    recordProps: {
      lvl0: {
        selectors: "h1",
      },
      lvl1: [
        "dl.method > dt > .descname",
        "dl.function > dt  > .descname",
        "h2",
      ],
      lvl2: "h3",
      lvl3: "h4",
      content: [
        "dl.method > dd",
        "dl.function > dd",
        // get how-to guides without breaking everything else:
        "section > section > section > dl:not(.class)",
      ],
      pageRank: "1",
    },
    indexHeadings: true,
  })
```

For each HTML tag, DocSearch checks if it (and its "parent" tags) match the recordProps. For
example, this matches:

```
<h1>Hello!</h1>
```

It becomes a lvl0 hierarchy record. All sibling tags which appear after this `h1` are considered
children of it. For example, the following page would generate two records, one at `lvl0` and one at
`lvl1`:

```
<h1>Hello!</h1>
<h2>Good bye!</h2>
```

Terminal records are only generated when the "content" CSS selector matches.

I manually iterated on this, checking one how-to guide page, one class reference page, and one VDS
reference page, until I found a set of CSS selectors that appeared to me to parse the page into a
reasonable hierarchy.

Once you're happy with the configuration, "Save" it. You can now eagerly request a crawl of the
website. Once the crawl is complete, as long as the new index doesn't have many fewer records, it
will immediately become active. Every search on hail.is will now use your new index.
