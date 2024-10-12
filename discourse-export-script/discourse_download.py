from argparse import ArgumentParser
from asyncio import gather, run, sleep
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from json import dump, loads
from re import sub
from typing import Any, Callable, Dict, List, Tuple, TypeVar

from aiohttp import ClientSession

strptime = datetime.strptime
strftime = datetime.strftime

# constants
POST_LINK_ID = "f4706281-cc60-4ff0-a0b6-b803683cc24b"
COMMENT_END_ID = "917f6034-2117-4a8c-bb42-b27fd7fb5e83"

# types
CallbackResponse = TypeVar("CallbackResponse")
Callback = Callable[[Dict[str, Any]], CallbackResponse]


@dataclass(frozen=True)
class DiscoursePost:
    id: int
    topic_id: int
    username: str
    created_at: str
    html: str


@dataclass(frozen=True)
class DiscourseTopic:
    id: int
    slug: str
    title: str
    html: str


# html parser
class DiscourseHTMLParser(HTMLParser):
    def __init__(self: "DiscourseHTMLParser") -> None:
        super().__init__()
        self.output_html = ""
        # relative file links starting with /
        self.relative_link = False
        # link previews and quotes
        self.aside = False
        self.aside_src = None
        self.aside_src_written = False
        self.aside_header = False
        self.aside_header_link = False
        self.aside_header_link_written = False
        # code blocks
        self.code_block_pre = False
        self.code_block_code = False
        # @ mentions
        self.mention = False

    def _decl_handler(self: "DiscourseHTMLParser", decl: str) -> None:
        self.output_html += f"<!{decl}>"

    def _ref_handler(self: "DiscourseHTMLParser", name: str) -> None:
        self.output_html += f"&{name};"

    def _write_starttag(self: "DiscourseHTMLParser", attrs: List[Tuple[str, str]], tag: str, suffix: str) -> None:
        attr_str_prefix = " " if len(attrs) > 0 else ""
        attr_str = " ".join([f'{key}="{value}"' for key, value in attrs])
        self.output_html += f"<{tag}{attr_str_prefix}{attr_str}{suffix}>"

    def _starttag_handler(suffix: str = "") -> None:
        def inner(self: "DiscourseHTMLParser", tag: str, attrs: List[Tuple[str, str]]) -> None:
            attr_dict = dict(attrs)
            if ((not self.aside) or self.aside_header) and tag == "a":
                if self.aside_header and not self.aside_header_link_written:
                    self.aside_header_link = True
                link = attr_dict.get("href", "")
                if "mention" in attr_dict.get("class", ""):
                    self.mention = True
                elif link.startswith("/"):
                    self.relative_link = True
                elif "https://discuss.hail.is/t/" in link:
                    slug = link.removeprefix("https://discuss.hail.is/t/").split("/")[0]
                    self.output_html += f'<a href="{POST_LINK_ID}/{slug}">'
                else:
                    self._write_starttag(attrs, tag, suffix)
            elif (
                self.aside
                and self.aside_src is None
                and (tag == "header" or (tag == "div" and "title" in attr_dict.get("class", "")))
            ):
                self.aside_header = True
                self.output_html += "\n"
            elif self.aside_header and tag == "blockquote":
                self.aside = False
                self.aside_header = False
                self.aside_header_link_written = False
                self._write_starttag(attrs, tag, suffix)
            elif self.aside_header and tag == "article":
                self.aside_header = False
                self.aside_header_link_written = False
            elif not self.aside:
                if tag == "aside":
                    self.aside = True
                    onebox_src = attr_dict.get("data-onebox-src", None)
                    if onebox_src is not None:
                        self.aside_src = onebox_src
                        self.output_html += f'\n<a href="{onebox_src}">'
                elif tag == "pre":
                    self.code_block_pre = True
                elif self.code_block_pre:
                    if tag == "code":
                        self.output_html += "\n\n```python\n"
                        self.code_block_code = True
                else:
                    self._write_starttag(attrs, tag, suffix)

        return inner

    handle_charref = _ref_handler
    handle_decl = _decl_handler
    handle_entityref = _ref_handler
    handle_startendtag = _starttag_handler(" /")
    handle_starttag = _starttag_handler()
    unknown_decl = _decl_handler

    def handle_comment(self: "DiscourseHTMLParser", data: str) -> None:
        self.output_html += f"<!--{data}-->"

    def handle_data(self: "DiscourseHTMLParser", data: str) -> None:
        if "https://discuss.hail.is/t/" in data:
            data = sub('https://discuss.hail.is/t/([A-Za-z0-9\\-]*?)', f'{POST_LINK_ID}/\\1', data)
        if self.mention:
            self.output_html += f'{data.partition("@")[2]}'
        elif self.aside_src is not None and not self.aside_src_written:
            self.output_html += self.aside_src
            self.aside_src_written = True
        elif (not self.aside) or self.aside_header_link:
            self.output_html += data

    def handle_endtag(self: "DiscourseHTMLParser", tag: str) -> None:
        if ((not self.aside) or self.aside_header) and tag == "a":
            if self.mention:
                self.mention = False
            elif self.relative_link:
                self.relative_link = False
            else:
                if self.aside_header_link:
                    self.aside_header_link = False
                    self.aside_header_link_written = True
                self.output_html += "</a>"
        elif tag == "aside":
            self.aside = False
            if self.aside_src is not None:
                self.output_html += "</a>\n"
                self.aside_src = None
                self.aside_src_written = False
                self.aside_header_link_written = True
        elif not self.aside:
            if tag == "pre":
                self.code_block_pre = False
            elif self.code_block_pre:
                if tag == "code":
                    self.output_html += "\n```\n\n"
                    self.code_block_code = False
            else:
                self.output_html += f"</{tag}>"

    def handle_pi(self: "DiscourseHTMLParser", data: str) -> None:
        self.output_html += f"<?{data}>"


# main script
async def main(discourse_page: int) -> None:
    async with ClientSession() as session:
        pages = await run_tasks(
            [parse_page(discourse_page, session)]
            # [parse_page(page, session) for page in range(discourse_page + 1)]
        )
        topics = await run_tasks([
            parse_topic(topic["id"], session) for page in pages for topic in page["topic_list"]["topics"]
        ])
        posts = await run_tasks([
            parse_post(post["id"], session) for topic in topics for post in topic["post_stream"]["posts"]
        ])

        topic_acc = {topic["id"]: {"fields": topic, "posts": []} for topic in topics}
        for post in posts:
            topic_acc[post.topic_id]["posts"].append(post)

        topics = []
        for topic_id, topic in topic_acc.items():
            if topic["fields"]["slug"] != "welcome-to-the-hail-community":
                topic_html = ""
                for idx, post in enumerate(topic["posts"]):
                    parser = DiscourseHTMLParser()
                    parser.feed(post.html)
                    topic_html += f"> [!NOTE]\n> The following post was exported from discuss.hail.is, a forum for asking questions about Hail which has since been deprecated.\n\n## ({strptime(post.created_at, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%b %d, %Y at %H:%M')}) {post.username} said:\n{parser.output_html} {COMMENT_END_ID if idx < (len(topic['posts']) - 1) else ''}"
                with open(f'./discourse-export/{topic["fields"]["id"]:04}_{topic["fields"]["slug"]}.json', 'w') as file:
                    dump(
                        {
                            "id": topic["fields"]["id"],
                            "slug": topic["fields"]["slug"],
                            "title": topic["fields"]["title"],
                            "html": topic_html,
                        },
                        file,
                    )


async def run_tasks(tasks):
    result = []
    while len(tasks) != 0:
        if len(tasks) > 5:
            result += await gather(*tasks[:4])
            tasks = tasks[4:]
            await sleep(2)
        else:
            result += await gather(*tasks)
            tasks = []
    return result


async def parse_page(discourse_page: int, session: ClientSession) -> None:
    async with session.get(f"https://discuss.hail.is/latest.json?page={discourse_page}") as response:
        return await response.json()


async def parse_topic(topic_id: int, session: ClientSession) -> None:
    async with session.get(f"https://discuss.hail.is/t/{topic_id}.json") as response:
        return loads(await response.read())


async def parse_post(post_id: int, session: ClientSession) -> None:
    async with session.get(f"https://discuss.hail.is/posts/{post_id}.json") as response:
        response_json = loads(await response.read())
        return DiscoursePost(
            response_json["id"],
            response_json["topic_id"],
            response_json["username"],
            response_json["created_at"],
            response_json["cooked"],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--page")
    args = parser.parse_args()
    run(main(int(args.page)))
