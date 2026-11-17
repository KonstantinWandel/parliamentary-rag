# Daily totals without addresses

`count_daily.py` turns the web server's access log into one row per day (`date,page_loads,visitors,searches,answers`)
in `/var/lib/plenarsuche-stats/daily.csv` and keeps no address. `visitors` is only the number of different addresses
that loaded the page, searched or asked for an answer on that day. Addresses in `/etc/plenarsuche-stats.exclude`
(the developer's own machines) are left out. A day is written once it is over and the log has no gap longer than two
hours inside it; existing rows are never changed. With `--cleanup`, rolled log files older than six days are deleted
as well, so the retention stated for the access log also holds for files that logrotate does not match
(`access-<timestamp>.log.gz`, written by Caddy itself).

Install: copy the script to `/opt/geolab/plenarsuche-stats/`, the two unit files to `/etc/systemd/system/`, list the
excluded addresses (one per line) in `/etc/plenarsuche-stats.exclude`, then
`systemctl daemon-reload && systemctl enable --now plenarsuche-stats.timer`. The timer runs at 00:35 Europe/Berlin.
Check without writing: `sudo python3 count_daily.py --dry-run`.

Read the totals: `ssh <host> cat /var/lib/plenarsuche-stats/daily.csv`.
