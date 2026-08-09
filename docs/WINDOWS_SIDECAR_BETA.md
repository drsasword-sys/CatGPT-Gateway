# Windows sidecar beta

The Windows sidecar is the local CatGPT provider bundled with App-Stran. It
does not require Docker, Xvfb, or noVNC. Chromium runs visibly on the user's
machine and stores its persistent profile under `%LOCALAPPDATA%\AppStran\CatGPT`.

Security and product constraints:

- the API binds to `127.0.0.1` only;
- App-Stran generates a random bearer token for each managed process;
- browser cookies and profile data never leave the user's machine;
- the server starts in `LOGIN_REQUIRED` recovery mode before first login;
- browser automation is experimental and may require updates when ChatGPT UI changes;
- the beta defaults to two lanes and must not be advertised as an official OpenAI API.

Build on Windows 11 or `windows-latest` GitHub Actions:

```powershell
.\scripts\build_windows_sidecar.ps1
```

Output:

```text
dist\CatGPTGateway-Windows-x64-beta.zip
```
