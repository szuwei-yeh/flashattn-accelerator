"""Run the four-state output-buffer test; VCS can return zero after $fatal."""
import re
import subprocess
import sys
from pathlib import Path

PASS_MARKER = ('RESULT: PASS four-state initialization, stale overwrite, reset, '
               'wrap, normalization')


def main():
    # Capture this invocation directly, so an old run.log cannot supply a PASS.
    log = Path('run.log')
    log.write_text('')
    result = subprocess.run(['./simv', '-no_save'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    output = result.stdout
    log.write_text(output)
    sys.stdout.write(output)
    failure = re.search(r'^\s*(?:Fatal|Error)(?:[:\s-])|\b(?:FAIL|FAILED|MISMATCH)\b',
                        output, re.IGNORECASE | re.MULTILINE)
    if result.returncode != 0 or failure or PASS_MARKER not in output.splitlines():
        print('VCS test rejected: require zero exit, explicit PASS, and no failure diagnostics.',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
