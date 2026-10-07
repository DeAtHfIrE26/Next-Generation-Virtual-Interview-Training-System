import { Alert } from "@/components/ui";

export default function PrivacyNotice() {
  return (
    <article className="mx-auto flex max-w-[720px] flex-col gap-6 px-5 py-16 text-[15px] leading-7 text-fg md:py-20">
      <Alert tone="warn">Draft for legal review. Not yet the operative privacy notice.</Alert>
      <header>
        <p className="text-[13px] font-medium text-accent">Legal</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9 sm:text-[36px] sm:leading-[44px]">Privacy notice (summary)</h1>
      </header>
      <p className="text-fg-muted">We process your account details, resume (with contact details removed before any AI processing), interview answers, and the signals listed below only to run practice interviews and give you feedback.</p>
      <ul className="flex list-disc flex-col gap-3 pl-5 text-fg-muted marker:text-fg-subtle">
        <li><strong className="font-semibold text-fg">Camera:</strong> analysed in your browser. We receive numbers (mouth movement, whether you are facing the screen, number of faces, whether a phone is visible), not video.</li>
        <li><strong className="font-semibold text-fg">Optional face and voice checks:</strong> stored as encrypted numeric templates, deleted after the retention period, when you withdraw consent, or when you delete your account.</li>
        <li><strong className="font-semibold text-fg">Recordings:</strong> answer audio is processed to transcribe and check your answer and is not kept unless you opt in.</li>
        <li><strong className="font-semibold text-fg">No emotion or personality inference.</strong> Feedback describes observable behaviour in this session.</li>
        <li><strong className="font-semibold text-fg">Your rights:</strong> download or delete everything from Privacy settings.</li>
      </ul>
      <p className="border-t border-line pt-6 text-[13px] leading-5 text-fg-subtle">The full draft is maintained in docs/legal/privacy-policy-draft.md.</p>
    </article>
  );
}
