# CayleyKernel in ChatGPT

Official OpenAI documentation checked **November 10, 2026** (America/Vancouver).

## Feasibility

**A private ChatGPT trial is feasible with a custom MCP plugin.** Expose one read-only tool over Streamable HTTP at `/mcp`. A user installs the plugin, selects it in a new ChatGPT Work chat with an @-mention, and pastes the two vectors. ChatGPT calls the tool with the arrays and can relay the structured product in the conversation. Custom UI is optional. [Plugin quickstart](https://developers.openai.com/plugins/quickstart) · [Build an MCP server](https://developers.openai.com/plugins/build/mcp-server)

GPT Actions are another documented integration for eligible custom GPTs. They use a separate REST API described by OpenAPI, require publicly trusted HTTPS with TLS 1.2 or later on port 443, and support no authentication, an API key, or OAuth. OpenAI is transitioning custom GPT workflows to plugins and custom actions need to be rebuilt during migration. Check the target workspace's GPT editing, sharing, and publishing eligibility before choosing that route. The Responses API can also call an MCP server from an application the owner builds. The proof of concept uses ChatGPT's custom MCP plugin route. [GPT Actions setup](https://developers.openai.com/api/docs/actions/getting-started) · [Action hosting](https://developers.openai.com/api/docs/actions/production) · [Action authentication](https://developers.openai.com/api/docs/actions/authentication) · [GPT workspace controls](https://learn.chatgpt.com/docs/enterprise/gpts-and-sharing) · [GPT migration](https://learn.chatgpt.com/docs/migrate-custom-gpts) · [Responses API MCP](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)

## Private testing and hosted trial

1. Use a ChatGPT account that can add custom MCP servers on the web. In a managed workspace, the owner or admin should check **Create plugins with MCPs** and **Use plugins**, plus sharing or workspace publishing permissions if teammates will participate. Directory review is for public distribution. [Add custom MCP server](https://developers.openai.com/api/docs/guides/custom-mcp-server) · [Workspace permissions](https://learn.chatgpt.com/docs/enterprise/apps-and-connectors) · [Remote MCP review](https://developers.openai.com/plugins/deploy/app-review)
2. Run the Python adapter on a CPU host with NumPy. For the private trial, connect a reachable HTTPS `/mcp` endpoint, a development HTTPS forwarding URL, or OpenAI Secure MCP Tunnel. For Secure MCP Tunnel, the owner needs a Platform tunnel ID, a runtime API key for `tunnel-client`, and the appropriate organization roles (Read + Manage to create, Read + Use to connect). Associate the tunnel with the target ChatGPT workspace as applicable. The normal HTTPS plugin connection and the local test do not require an OpenAI API key. [Connect and test](https://developers.openai.com/plugins/deploy/connect-chatgpt) · [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
3. In ChatGPT Plugins, choose **Add custom MCP server**, set the URL or tunnel, select **No authentication**, and create and install the personal plugin. Anonymous access suits this stateless computation on submitted operands. ChatGPT also supports OAuth and mixed anonymous/OAuth tools. If access to a reachable endpoint must be restricted to specific users, use the private tunnel where appropriate or add OAuth 2.1 and verify access tokens server-side. [Add custom MCP server](https://developers.openai.com/api/docs/guides/custom-mcp-server) · [MCP authentication](https://developers.openai.com/plugins/build/auth)
4. For a reachable host, configure the proxy and allowed host names for Streamable HTTP, set request-body and rate limits and timeouts, and keep vectors out of operational logs. Recheck latency and concurrent load on the target host. In ChatGPT, try explicit invocation with the octonion example below, then check that the tool receives coefficients in order, presents the returned vector accurately at the 256-coefficient boundary, and handles mismatches and over-limit requests. Hosting and model invocation remain unverified. [Deploy an MCP endpoint](https://developers.openai.com/plugins/build/mcp-server) · [Connect and test](https://developers.openai.com/plugins/deploy/connect-chatgpt)

## Public distribution

- Use a stable, publicly reachable HTTPS Streamable HTTP endpoint and keep it available for review. Temporary and Secure MCP tunnels are trial options. The separately documented desktop-only submission route for local MCP requires approved-partner status. [Deploy an MCP endpoint](https://developers.openai.com/plugins/build/mcp-server) · [Desktop-only submissions](https://developers.openai.com/plugins/deploy/desktop-only)
- Choose the owning Platform organization and project. An organization owner or a member with **Apps Management Write** submits under a verified individual or business identity. Upload a plugin ZIP **With MCP**, complete the domain challenge at the portal-specified `/.well-known/openai-apps-challenge` URL, and resolve tool scan findings. [Package a plugin](https://developers.openai.com/plugins/build/plugins) · [Upload and submit](https://developers.openai.com/plugins/deploy/submission)
- Prepare listing metadata, icon, supported countries and localization, HTTPS website/support/privacy/terms URLs, a demo recording, five positive and three negative review cases, release notes, and policy attestations. Supply dedicated reviewer credentials if OAuth is used. The tool explicitly declares read-only, non-destructive and closed-world behavior. The guidelines say annotation justifications are no longer required, while the submission validation reference still lists them. Check the live portal when preparing a submission. [Upload and submit](https://developers.openai.com/plugins/deploy/submission) · [Submission validation](https://developers.openai.com/plugins/deploy/submission-errors) · [Plugin guidelines](https://developers.openai.com/plugins/plugin-guidelines)
- Submit the finished plugin for review and publish after approval. Directory eligibility is unverified. The guidelines call for a complete, reliable workflow that adds value beyond built-in capabilities. A public submission would need to demonstrate the value of reproducible products in CayleyKernel's basis convention. [Plugin guidelines](https://developers.openai.com/plugins/plugin-guidelines) · [Remote MCP review](https://developers.openai.com/plugins/deploy/app-review)

## Operation and CPU cost

`cayley_dickson_multiply(left, right)` computes **left × right**. Each argument is a flat JSON array of finite numbers in CayleyKernel basis order `[e0, e1, …]`, with the scalar coefficient first. Both lengths must match and be one of `1, 2, 4, 8, 16, 32, 64, 128, 256`. The result is a structured object with one `product` array of the same length and order. Arithmetic and returned values use binary64, including ordinary rounding and underflow.

The underlying [CPU wrapper](../../CayleyKernel/_cpu.py) takes one-dimensional array-likes convertible to NumPy float64, rejects complex dtypes and complex objects, and returns a one-dimensional float64 NumPy array. Shape and length checks cover scalars, nested or batched arrays, empty vectors, unequal lengths, and lengths that are not powers of two. The library conversion can accept some coercible values such as numeric strings or booleans. The tool schema uses strict finite numeric coefficients and rejects strings, booleans, nulls, nested arrays, non-finite values and numbers outside the finite binary64 range. Validation failures are MCP tool errors. If finite operands produce any non-finite coefficient, the whole call fails with a tool error. The adapter uses the optional Rust extension when selected and available, with the Python implementation available via `CAYLEY_KERNEL_CPU_BACKEND=python`. Both paths use NumPy at the public boundary and run on CPU.

Dense multiplication evaluates `N²` terms. I timed `cpu_multiply` with the Python backend on an AMD Ryzen 9 7900X, CPython 3.14.3, and NumPy 2.4.1. Each size used two seeded, dense random lists (uniform on [-1, 1]), one warm-up, and nine serial wall-clock measurements including NumPy input conversion. Times exclude MCP and network overhead.

| Dimension | Median (ms) | Slowest of 9 (ms) |
| ---: | ---: | ---: |
| 8 | 0.026 | 0.030 |
| 64 | 1.615 | 1.732 |
| 128 | 6.228 | 6.500 |
| 256 | 25.415 | 26.524 |
| 512 | 91.130 | 93.161 |
| 1024 | 375.090 | 378.863 |

The initial limit of 256 bounds a call at 65,536 terms and left headroom on the slower backend relative to the next doubling. The tool schema enforces the cap before multiplication. These measurements are local single-call timings.

## Run and verify locally

From the repository root, use an isolated Python environment with the dependencies below. The adapter lives under `examples/` and is outside the packaged library API.

```bash
python3 -m venv /tmp/cayley-mcp-demo
source /tmp/cayley-mcp-demo/bin/activate
python -m pip install 'mcp==1.30.0' numpy pytest
CAYLEY_KERNEL_CPU_BACKEND=python python -m pytest -q examples/chatgpt_mcp/test_server.py
CAYLEY_KERNEL_CPU_BACKEND=python python -m examples.chatgpt_mcp.server
```

The server binds to `127.0.0.1:8000`. To try it interactively, open MCP Inspector with `npx @modelcontextprotocol/inspector`, select Streamable HTTP, and connect to `http://127.0.0.1:8000/mcp`. Call `cayley_dickson_multiply` with:

```json
{"left":[0,1,0,0,0,0,0,0],"right":[0,0,1,0,0,0,0,0]}
```

The expected result is `{"product":[0,0,0,1,0,0,0,0]}` because `e1 * e2 = e3` in this convention.

The standalone server returned that product over loopback. All 14 MCP tests passed on the Python fallback and again with the Rust extension (0.1.0). The 66 existing CPU backend tests also passed with the Rust extension installed. Coverage includes discovery, the octonion result, the 256-coefficient boundary, mismatched lengths, 512-coefficient rejection, invalid shapes and types, non-finite inputs, and product overflow.
