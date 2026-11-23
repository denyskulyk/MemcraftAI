from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MEMCRAFT_", extra="ignore")

    database_url: str = "postgresql+psycopg://memcraft@127.0.0.1:5432/memcraft"
    shared_dir: Path = Path("data/shared")
    # None in the web container, which does not see this folder.
    private_dir: Path | None = None
    # Names the browser may use, separated by commas. Add the PC's Tailscale name.
    allowed_hosts: str = "localhost,127.0.0.1"
    # This Tailscale login becomes the admin on its first visit. Empty: nobody does.
    admin_login: str = ""
    # The address people open, such as https://pc.tailnet.ts.net. Passkeys are bound to its
    # host name, so changing it means enrolling the admin passkey again.
    public_origin: str = "http://localhost:8000"

    @property
    def passkey_rp_id(self) -> str:
        host = urlsplit(self.public_origin).hostname
        if not host:
            raise ValueError("MEMCRAFT_PUBLIC_ORIGIN must be an address like https://name")
        return host

    @property
    def allowed_host_list(self) -> list[str]:
        hosts = [host.strip().lower() for host in self.allowed_hosts.split(",") if host.strip()]
        if any("*" in host for host in hosts):
            raise ValueError("MEMCRAFT_ALLOWED_HOSTS must not contain wildcards")
        return hosts


@lru_cache
def get_settings() -> Settings:
    return Settings()
