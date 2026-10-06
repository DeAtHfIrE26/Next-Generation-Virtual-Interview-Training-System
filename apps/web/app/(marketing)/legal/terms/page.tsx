import { Alert } from "@/components/ui";

export default function Terms() {
  return (
    <article className="mx-auto flex max-w-[720px] flex-col gap-6 px-5 py-16 text-[15px] leading-7 text-fg md:py-20">
      <Alert tone="warn">Draft for legal review.</Alert>
      <header>
        <p className="text-[13px] font-medium text-accent">Legal</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9 sm:text-[36px] sm:leading-[44px]">Terms (summary)</h1>
      </header>
      <p className="text-fg-muted">The AI Interview Coach is a practice tool for candidates preparing their own interviews. It must not be used to make or support hiring decisions about other people. Feedback is generated automatically, may be wrong, and is labelled experimental until validated.</p>
    </article>
  );
}
