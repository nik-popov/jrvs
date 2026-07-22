/**
 * GitHub helpers for the JRVS → coding-agent channel.
 *
 * request_feature files an issue on the JRVS repo (labeled "jrvs-request")
 * and assigns it to the Copilot coding agent, which implements it and opens
 * a pull request. Plain fetch against api.github.com — no SDK.
 */

const API = "https://api.github.com";

export interface GitHubEnv {
  GITHUB_TOKEN?: string;
  /** "owner/repo" — where JRVS files its own feature requests. */
  GITHUB_REPO?: string;
}

export const FEATURE_LABEL = "jrvs-request";

/** Login of the Copilot coding agent bot in suggestedActors. */
const COPILOT_LOGIN = "copilot-swe-agent";

function ghHeaders(token: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "jrvs-agent",
    "X-GitHub-Api-Version": "2022-11-28"
  };
}

function repoParts(env: GitHubEnv): { token: string; owner: string; repo: string } {
  if (!env.GITHUB_TOKEN) {
    throw new Error(
      "GitHub is not configured — set the GITHUB_TOKEN secret to enable feature requests."
    );
  }
  const [owner, repo] = (env.GITHUB_REPO ?? "").split("/");
  if (!owner || !repo) {
    throw new Error('GITHUB_REPO must be set to "owner/repo" in wrangler.jsonc vars.');
  }
  return { token: env.GITHUB_TOKEN, owner, repo };
}

async function graphql<T>(
  token: string,
  query: string,
  variables: Record<string, unknown>
): Promise<T> {
  const res = await fetch(`${API}/graphql`, {
    method: "POST",
    headers: ghHeaders(token),
    body: JSON.stringify({ query, variables })
  });
  if (!res.ok) {
    throw new Error(`GitHub GraphQL ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  const data = (await res.json()) as { data?: T; errors?: { message: string }[] };
  if (data.errors?.length) {
    throw new Error(`GitHub GraphQL: ${data.errors[0].message}`);
  }
  if (!data.data) throw new Error("GitHub GraphQL: empty response");
  return data.data;
}

/**
 * Assign the Copilot coding agent to an issue. Assignment uses GraphQL:
 * find the bot among the repo's suggested assignees, then replace assignees.
 */
async function assignCopilot(
  token: string,
  owner: string,
  repo: string,
  issueNodeId: string
): Promise<{ ok: boolean; error?: string }> {
  try {
    const found = await graphql<{
      repository: {
        suggestedActors: { nodes: { login: string; id?: string }[] };
      };
    }>(
      token,
      `query($owner: String!, $name: String!) {
        repository(owner: $owner, name: $name) {
          suggestedActors(capabilities: [CAN_BE_ASSIGNED], first: 100) {
            nodes { login ... on Bot { id } ... on User { id } }
          }
        }
      }`,
      { owner, name: repo }
    );
    const bot = found.repository.suggestedActors.nodes.find(
      (n) => n.login === COPILOT_LOGIN && n.id
    );
    if (!bot?.id) {
      return {
        ok: false,
        error: "Copilot coding agent is not an assignable actor on this repository."
      };
    }
    await graphql(
      token,
      `mutation($assignableId: ID!, $actorIds: [ID!]!) {
        replaceActorsForAssignable(input: { assignableId: $assignableId, actorIds: $actorIds }) {
          assignable { ... on Issue { number } }
        }
      }`,
      { assignableId: issueNodeId, actorIds: [bot.id] }
    );
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export interface FeatureIssue {
  number: number;
  url: string;
  assigned: boolean;
  assign_error?: string;
}

/** Create a labeled issue and (best-effort) hand it to the coding agent. */
export async function createFeatureIssue(
  env: GitHubEnv,
  title: string,
  body: string
): Promise<FeatureIssue> {
  const { token, owner, repo } = repoParts(env);
  const res = await fetch(`${API}/repos/${owner}/${repo}/issues`, {
    method: "POST",
    headers: ghHeaders(token),
    body: JSON.stringify({ title, body, labels: [FEATURE_LABEL] })
  });
  if (!res.ok) {
    throw new Error(`GitHub API ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  const issue = (await res.json()) as {
    number: number;
    html_url: string;
    node_id: string;
  };
  const assign = await assignCopilot(token, owner, repo, issue.node_id);
  return {
    number: issue.number,
    url: issue.html_url,
    assigned: assign.ok,
    ...(assign.error ? { assign_error: assign.error } : {})
  };
}

export interface FeatureRequestSummary {
  number: number;
  title: string;
  state: string;
  assigned_to: string[];
  url: string;
  created_at: string;
}

/** Recent feature requests JRVS has filed, newest first. */
export async function listFeatureRequests(
  env: GitHubEnv,
  max = 10
): Promise<FeatureRequestSummary[]> {
  const { token, owner, repo } = repoParts(env);
  const u = new URL(`${API}/repos/${owner}/${repo}/issues`);
  u.searchParams.set("labels", FEATURE_LABEL);
  u.searchParams.set("state", "all");
  u.searchParams.set("sort", "created");
  u.searchParams.set("direction", "desc");
  u.searchParams.set("per_page", String(max));
  const res = await fetch(u.toString(), { headers: ghHeaders(token) });
  if (!res.ok) {
    throw new Error(`GitHub API ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  const items = (await res.json()) as {
    number: number;
    title: string;
    state: string;
    html_url: string;
    created_at: string;
    assignees?: { login: string }[];
    pull_request?: unknown;
  }[];
  return items
    .filter((i) => !i.pull_request)
    .map((i) => ({
      number: i.number,
      title: i.title,
      state: i.state,
      assigned_to: (i.assignees ?? []).map((a) => a.login),
      url: i.html_url,
      created_at: i.created_at
    }));
}
