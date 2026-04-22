---
name: verify-links-before-sending
description: Before sending any generated URL, tunnel, preview, or endpoint to the user, verify that it is reachable and returns the expected response.
version: 1.0.0
---

# Verify Links Before Sending

Never hand the user a URL you have not checked yourself.

## When to Use
- You are about to send a local preview URL, tunnel URL, webhook URL, API endpoint, download URL, or browser-generated link
- You created or modified a service that should now be reachable from a URL
- The user asks whether a generated endpoint or tunnel works

## Deterministic First
- Use deterministic verification, not intuition.
- Check the exact URL with `curl`, a browser tool, or another direct network probe before sending it.
- If the endpoint should return structured data, inspect the actual status code and payload shape.
- If the endpoint is local or tunneled, verify both process readiness and URL reachability.

## Workflow
1. Identify the exact URL you plan to send.
2. Probe it directly with the most appropriate deterministic tool:
   - `curl -I` or `curl -sS` for HTTP endpoints
   - browser navigation when visual verification matters
   - a purpose-built health endpoint if one exists
3. Confirm the expected status code and content are returned.
4. If the endpoint fails, fix the issue first or tell the user it is not ready.
5. Only send the link after a successful check.

## Verification
- Include the fact that you checked the URL before sending it.
- For HTTP endpoints, verify at minimum that the URL is reachable and not returning an obvious error.
- For tunnels, verify the public URL itself, not just the local process.
- If the link is expected to be authenticated or interactive, note any assumptions or access requirements.
