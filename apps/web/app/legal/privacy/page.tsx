export default function PrivacyNotice() {
  return (
    <article className="stack" style={{ maxWidth: 760 }}>
      <p className="notice">Draft for legal review. Not yet the operative privacy notice.</p>
      <h1>Privacy notice (summary)</h1>
      <p>We process your account details, resume (with contact details removed before any AI processing), interview answers, and the signals listed below only to run practice interviews and give you feedback.</p>
      <ul>
        <li><strong>Camera:</strong> analysed in your browser. We receive numbers (mouth movement, whether you are facing the screen, number of faces, whether a phone is visible), not video.</li>
        <li><strong>Optional face and voice checks:</strong> stored as encrypted numeric templates, deleted after the retention period, when you withdraw consent, or when you delete your account.</li>
        <li><strong>Recordings:</strong> answer audio is processed to transcribe and check your answer and is not kept unless you opt in.</li>
        <li><strong>No emotion or personality inference.</strong> Feedback describes observable behaviour in this session.</li>
        <li><strong>Your rights:</strong> download or delete everything from Privacy settings.</li>
      </ul>
      <p className="small muted">The full draft is maintained in docs/legal/privacy-policy-draft.md.</p>
    </article>
  );
}
