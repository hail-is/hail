from argparse import ArgumentParser
from asyncio import gather, run, sleep
from datetime import datetime
from json import dumps, load, loads
from os import listdir
from os.path import isfile, join
from re import findall, sub
from shutil import move

from aiohttp import ClientSession

now = datetime.now
fromtimestamp = datetime.fromtimestamp

# constants
POST_LINK_ID = "f4706281-cc60-4ff0-a0b6-b803683cc24b"
COMMENT_END_ID = "917f6034-2117-4a8c-bb42-b27fd7fb5e83"


# TODO i think the numbering is off?
async def main(github_issue_number: int, github_token: str) -> None:
    links = {}
    for idx, filename in enumerate(sorted(listdir("./discourse-export"))):
        if isfile(join("./discourse-export", filename)):
            id, rest = filename.split("_")
            slug, _ = rest.split(".")
            with open(f"./discourse-export/{id}_{slug}.json", "r") as file:
                links[slug] = {
                    "id": id,
                    "idx": idx,
                    "dests": set([slug for slug in findall(f'{POST_LINK_ID}/([A-Za-z0-9\\-]*?)\\\\"', file.read())]),
                }
    for slug, data in links.items():
        for dest in data["dests"]:
            dest_data = links.get(dest, None)
            if dest_data is None:
                print(
                    f"broken link: {slug}->{dest} (https://github.com/hail-is/hail/issues/{github_issue_number + data['idx']})"
                )
            else:
                print(
                    f"link: {slug} (https://github.com/hail-is/hail/issues/{github_issue_number + data['idx']}) -> {dest} (https://github.com/hail-is/hail/issues/{github_issue_number + dest_data['idx']})"
                )
                with open(f"./discourse-export/{data['id']}_{slug}.json", "r") as file:
                    json = sub(
                        f'{POST_LINK_ID}/{dest}"',
                        f"https://github.com/hail-is/hail/issues/{github_issue_number + dest_data['idx']}\\\"",
                        file.read(),
                    )
                with open(f"./discourse-export/{data['id']}_{slug}.json", "w") as file:
                    file.write(json)
    async with ClientSession() as session:
        for issue in sorted([{"slug": slug, **data} for slug, data in links.items()], key=lambda x: x["idx"]):
            with open(f"./discourse-export/{issue['id']}_{issue['slug']}.json", "r") as file:
                topic = load(file)
            discussion_id, label_applied, comment_idx = [None, False, 0]
            comments = topic["html"].split(COMMENT_END_ID)
            discussion_html = comments[0]
            rest_comments = comments[1:]
            while discussion_id is None:
                discussion_id = next(
                    iter(await gather(create_discussion(discussion_html, topic["title"], session, github_token)))
                )
            while not label_applied:
                label_applied = next(iter(await gather(apply_label(discussion_id, session, github_token))))
            while comment_idx < (len(rest_comments)):
                comment_idx = next(
                    iter(
                        await gather(
                            add_comment(comment_idx, rest_comments[comment_idx], discussion_id, session, github_token)
                        )
                    )
                )
            move(
                f"./discourse-export/{issue['id']}_{issue['slug']}.json",
                f"./uploaded/{issue['id']}_{issue['slug']}.json",
            )


async def add_comment(comment_idx, comment_html, discussion_id, session, github_token):
    comment_query = f"""
    mutation {{
      addDiscussionComment (
        input: {{
          discussionId: "{discussion_id}"
          body: {dumps(comment_html)}
        }}
      ) {{
        comment {{
            id
        }}
      }}
    }}
    """
    async with session.post(
        "https://api.github.com/graphql",
        json={"query": comment_query},
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "Content-Type": "application/json; charset=utf-8",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    ) as comment_response:
        comment_response_json = loads(await comment_response.read())
        if comment_response_json.get("errors", None) is not None:
            print(comment_response_json)
            await handle_error(comment_response.headers)
            return comment_idx
    return comment_idx + 1


async def apply_label(discussion_id, session, github_token):
    label_query = f"""
    mutation {{
      addLabelsToLabelable (
        input: {{
          labelableId: "{discussion_id}"
          labelIds: ["LA_kwDOKFqpFc8AAAABajc5aQ"]
        }}
      ) {{
        labelable {{
          labels {{
            totalCount
          }}
        }}
      }}
    }}
    """
    async with session.post(
        "https://api.github.com/graphql",
        json={"query": label_query},
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "Content-Type": "application/json; charset=utf-8",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    ) as label_response:
        label_response_json = loads(await label_response.read())
        if label_response_json.get("errors", None) is not None:
            print(label_response_json)
            await handle_error(label_response.headers)
            return False
    return True


async def create_discussion(discussion_html, discussion_title, session: ClientSession, github_token: str) -> bool:
    discussion_query = f"""
    mutation {{
      createDiscussion(
        input: {{
          repositoryId: "R_kgDOKFqpFQ",
          categoryId: "DIC_kwDOKFqpFc4CYhFv",
          body: {dumps(discussion_html)},
          title: "{discussion_title}"
        }}
      ) {{
        discussion {{
          id
        }}
      }}
    }}
    """
    async with session.post(
        "https://api.github.com/graphql",
        json={"query": discussion_query},
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "Content-Type": "application/json; charset=utf-8",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    ) as discussion_response:
        discussion_response_json = loads(await discussion_response.read())
        if discussion_response_json.get("errors", None) is not None:
            print(discussion_response_json)
            await handle_error(discussion_response.headers)
            return None
        return discussion_response_json["data"]["createDiscussion"]["discussion"]["id"]


async def handle_error(headers):
    retry_time = fromtimestamp(int(headers.get("X-RateLimit-Reset")))
    if retry_time > now():
        print(f"Retry time is {retry_time - now()}; waiting for 1 minute...")
        await sleep(60)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--github_issue_number")
    parser.add_argument("--github_token")
    args = parser.parse_args()
    run(main(int(args.github_issue_number), args.github_token))


# TODO change ids to match hail repo using following queries

# query {
#  repository (name: "test-process", owner: "iris-garden") {
#    id
#    name
#  }
# }

# query {
#  repository (name: "test-process", owner: "iris-garden") {
#    discussionCategories (first: 100) {
#      edges {
#        node {
#          name
#          id
#        }
#      }
#    }
#  }
# }

# query {
#  repository (name: "test-process", owner: "iris-garden") {
#    labels (first: 100) {
#      edges {
#        node {
#          id
#          name
#        }
#      }
#    }
#  }
# }
