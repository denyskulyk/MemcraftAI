# Postgres in Podman

Run everything from `C:\Source\MemcraftAI`. Each script works in PowerShell 7 and Windows PowerShell 5.1
(`powershell` instead of `pwsh`). No admin rights needed.

1. Start the Podman machine. The scripts below never do this for you.

```powershell
podman machine start
```

2. Create `.env`, the secret and the container, and wait until it is healthy. The first run downloads the image.

```powershell
pwsh -ExecutionPolicy Bypass -File .\deploy\postgres.ps1
```

3. Create the tables.

```powershell
pwsh -ExecutionPolicy Bypass -File .\deploy\migrate.ps1
```

4. Make a backup in `backups\`. The newest 14 are kept; change that with `-Keep`.

```powershell
pwsh -ExecutionPolicy Bypass -File .\deploy\backup.ps1
```

5. Test a backup by restoring it into a new database, `memcraft_restore_test`.

```powershell
pwsh -ExecutionPolicy Bypass -File .\deploy\restore.ps1 -Dump .\backups\memcraft-YYYYMMDD-HHmmss.dump
```

## Good to know

- The existing logon task `LocalAI podman machine` starts the Podman machine. Starting it before logon needs admin
  rights and is handled separately by `deploy\admin-setup.ps1`.
- The password is in `.env` and the Podman secret `memcraft-pg-password`. Changing one does not change the other or
  the database. Keep a copy of `.env`: without it `postgres.ps1` stops rather than invent a second password.
- Changed container settings need `podman rm -f memcraft-postgres` and `postgres.ps1` again. The data stays in the
  volume `memcraft-pgdata`. Do not remove that volume.
- Postgres does not start with the machine, by choice. After a restart, run `postgres.ps1` again; it starts the existing container.
- A failed restore can leave an empty new database. Drop it, or use another `-Database` name.
- `backups\` is on the same PC. Copy it elsewhere too.

## Tested on 2026-11-23

Container, secret, health, migration, backup and restore into a new database, and the full test suite
(`MEMCRAFT_TEST_DATABASE_URL` pointing at `memcraft_test`). Postgres answers on 127.0.0.1 and not on the LAN or
Tailscale address of this PC, checked from this PC.

Not tested: coming back after a Windows restart, and access from another device.
