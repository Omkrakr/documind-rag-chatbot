"""
Generates the ER diagram for DocuMind's relational schema.
"""
from graphviz import Digraph

g = Digraph("er_diagram", format="png")
g.attr(rankdir="LR", bgcolor="white", fontname="Helvetica", nodesep="0.6", ranksep="0.9", pad="0.4")
g.attr(label="DocuMind \u2014 Database Schema (Entity-Relationship)", labelloc="t",
       fontsize="16", fontcolor="#1F2A38", fontname="Helvetica-Bold")
g.attr("node", fontname="Helvetica", fontsize="10", shape="none")
g.attr("edge", fontname="Helvetica", fontsize="9", color="#5A6472", arrowsize="0.7")


def entity(name, pk, fields):
    rows = "".join(
        f'<tr><td align="left" port="{f.split(":")[0].strip().lstrip("*")}">{f}</td></tr>'
        for f in fields
    )
    label = f'''<
    <table border="1" cellborder="0" cellspacing="0" cellpadding="6" bgcolor="#FAFBFD" color="#5C7AA8">
      <tr><td bgcolor="#3F6694" align="center"><font color="white"><b>{name}</b></font></td></tr>
      <tr><td align="left"><b>PK&#160;&#160;{pk}</b></td></tr>
      {rows}
    </table>>'''
    g.node(name, label=label)


entity("users", "id (uuid)", ["name: string(120)", "email: string(255) UNIQUE", "created_at: datetime"])
entity("documents", "id (uuid)", ["FK&#160;&#160;user_id &#8594; users.id", "filename: string(255)",
                                   "status: enum(pending/processing/ready/failed)",
                                   "chunk_count: int", "uploaded_at: datetime"])
entity("document_chunks", "id (uuid)", ["FK&#160;&#160;document_id &#8594; documents.id",
                                         "chunk_index: int", "text: text"])
entity("conversations", "id (uuid)", ["FK&#160;&#160;user_id &#8594; users.id",
                                       "title: string(255)", "created_at: datetime"])
entity("messages", "id (uuid)", ["FK&#160;&#160;conversation_id &#8594; conversations.id",
                                  "role: enum(user/assistant)", "content: text",
                                  "sources: text (JSON list of chunk_ids)",
                                  "created_at: datetime"])

g.edge("users", "documents", label="1 .. *", color="#5C7AA8")
g.edge("documents", "document_chunks", label="1 .. *", color="#5C7AA8")
g.edge("users", "conversations", label="1 .. *", color="#5C7AA8")
g.edge("conversations", "messages", label="1 .. *", color="#5C7AA8")

g.render("/home/claude/documind/docs/diagrams/er_diagram", cleanup=True)
print("done")
