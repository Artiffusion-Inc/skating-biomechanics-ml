#!/usr/bin/env bash
# Create Resend hosted email templates.
# Requires: RESEND_API_KEY env var (or resend login --key)
# Usage: bash scripts/upload_email_templates.sh

set -euo pipefail

TEMPLATES_DIR="$(dirname "$0")/email_templates"

create_template() {
  local name="$1" alias="$2" subject="$3" html_file="$4" vars=("${@:5}")

  local cmd="resend templates create --name \"$name\" --alias \"$alias\" --subject \"$subject\" --html-file \"$html_file\""

  for var in "${vars[@]}"; do
    cmd="$cmd --var $var"
  done

  echo "Creating: $name ($alias)"
  eval "$cmd"
  echo "---"
}

# Password reset — Russian
create_template \
  "Password Reset (RU)" "password-reset-ru" \
  "Сброс пароля — SkateLab" \
  "$TEMPLATES_DIR/password-reset-ru.html" \
  "RESET_URL:string"

# Password reset — English
create_template \
  "Password Reset (EN)" "password-reset-en" \
  "Password Reset — SkateLab" \
  "$TEMPLATES_DIR/password-reset-en.html" \
  "RESET_URL:string"

# Email verification — Russian
create_template \
  "Email Verification (RU)" "email-verification-ru" \
  "Подтверждение email — SkateLab" \
  "$TEMPLATES_DIR/email-verification-ru.html" \
  "VERIFY_URL:string"

# Email verification — English
create_template \
  "Email Verification (EN)" "email-verification-en" \
  "Email Verification — SkateLab" \
  "$TEMPLATES_DIR/email-verification-en.html" \
  "VERIFY_URL:string"

# Coaching invite — Russian
create_template \
  "Coaching Invite (RU)" "coaching-invite-ru" \
  "Приглашение на SkateLab" \
  "$TEMPLATES_DIR/coaching-invite-ru.html" \
  "INVITER_NAME:string" "CONNECTION_TYPE:string" "DASHBOARD_URL:string"

# Coaching invite — English
create_template \
  "Coaching Invite (EN)" "coaching-invite-en" \
  "Invitation on SkateLab" \
  "$TEMPLATES_DIR/coaching-invite-en.html" \
  "INVITER_NAME:string" "CONNECTION_TYPE:string" "DASHBOARD_URL:string"

echo ""
echo "Templates created as drafts. Publish each one:"
echo "  resend templates publish <template-id>"
echo ""
echo "Or list all:"
echo "  resend templates list"