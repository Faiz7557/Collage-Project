{{-- partials/update-profile-information-form.blade.php --}}
<section class="space-y-6">
    <header class="profile-section-header">
        <h2 class="text-lg font-medium text-white">
            {{ __("Update your account's profile information and email address.") }}
        </h2>
    </header>

    <form id="send-verification" method="post" action="{{ route('verification.send') }}">
        @csrf
    </form>

    <form method="post" action="{{ route('profile.update') }}" class="mt-6 space-y-6">
        @csrf
        @method('patch')

        <!-- Foto Profil (Tambahan Upgrade) -->
        <div>
            <x-input-label for="foto" :value="__('Foto Profil (Opsional)')" />
            <div class="file-upload-wrapper">
                <input type="file" id="foto" name="foto" accept="image/*" class="form-input-custom w-full">
                <label for="foto" class="file-upload-label">
                    <i class="fas fa-camera me-2"></i>Pilih Foto
                </label>
            </div>
            <x-input-error class="mt-2" :messages="$errors->get('foto')" />
            @if(auth()->user()->foto)
                <p class="text-light small mt-1">Foto saat ini: {{ auth()->user()->foto }}</p>
            @endif
        </div>

        <div>
            <x-input-label for="name" :value="__('Name')" />
            <x-text-input id="name" name="name" type="text" class="mt-1 block w-full form-input-custom" :value="old('name', $user->name)" required autofocus autocomplete="name" />
            <x-input-error class="mt-2" :messages="$errors->get('name')" />
        </div>

        <div>
            <x-input-label for="email" :value="__('Email')" />
            <x-text-input id="email" name="email" type="email" class="mt-1 block w-full form-input-custom" :value="old('email', $user->email)" required autocomplete="username" />
            <x-input-error class="mt-2" :messages="$errors->get('email')" />

            @if (Auth::user() instanceof \Illuminate\Contracts\Auth\MustVerifyEmail && ! Auth::user()->hasVerifiedEmail())
                <div class="verification-alert mt-3 p-3 bg-white/20 rounded-lg">
                    <p class="text-light mb-2">
                        <i class="fas fa-exclamation-circle me-1 text-warning"></i>
                        {{ __('Your email address is unverified.') }}
                    </p>
                    <button form="send-verification" class="btn-verify-email">
                        {{ __('Click here to re-send the verification email.') }}
                    </button>
                    @if (session('status') === 'verification-link-sent')
                        <p class="text-success mt-2">
                            <i class="fas fa-check me-1"></i>
                            {{ __('A new verification link has been sent to your email address.') }}
                        </p>
                    @endif
                </div>
            @endif
        </div>

        <div class="flex items-center gap-4">
            <x-primary-button class="btn-save-profile">
                <i class="fas fa-save me-2"></i>{{ __('Save') }}
            </x-primary-button>

            @if (session('status') === 'profile-updated')
                <p
                    x-data="{ show: true }"
                    x-show="show"
                    x-transition
                    x-init="setTimeout(() => show = false, 3000)"
                    class="text-success font-medium"
                >
                    <i class="fas fa-check-circle me-1"></i>{{ __('Saved.') }}
                </p>
            @endif
        </div>
    </form>
</section>